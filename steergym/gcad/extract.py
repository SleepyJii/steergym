"""Fit-time extraction (paper Sec. 4, Eq. 6-9; upstream core/generate_vec.py's
`get_prompt_attn_delta_p_and_r` + scripts/generate_vec_prompt_attn_k.py's
`compute_k_pos_and_avg_demand`).

Crop window S (Eq. 7): upstream crops to the SYSTEM-PROMPT turn of a
persona-vectors chat string. steergym rows carry the instruction folded
into `prompt` ahead of the question (personavec `fold`), so S = the
rendered prompt up to where `meta["question"]` begins (template prefix +
instruction, upstream's analog); rows with no `meta["question"]` suffix
crop to the whole prompt -- see the package README. Requires `attn_implementation="eager"`: SDPA/FlashAttn don't
expose `attn_weights` even with `output_attentions=True` (upstream's own
comment, `generate_vec.py:252-253`). Unbatched (one forward per row, no
padding), matching upstream's own loop -- fit-time scale only, not a
generation hot path. Torch imported lazily."""


def layer_dims(model):
    cfg = model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads  # gemma-2 declares head_dim != hidden/heads
    return num_heads, num_kv_heads, num_heads // num_kv_heads, head_dim


def crop_length(tokenizer, prompt_text, prompt_len, question):
    """Tokens of the rendered prompt before `question`'s last occurrence;
    `prompt_len` when there is no question to split off."""
    if not question:
        return prompt_len
    idx = prompt_text.rfind(question)
    if idx <= 0:
        return prompt_len
    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    return sum(1 for start, _ in enc["offset_mapping"] if start < idx)


def contrast_rows(samples, concept, tokenizer, suffix_length):
    """[(text, prompt_len, crop_len), ...] for each of the concept's positive/negative
    rows: `text` is `binarize_text`'s [user, assistant] chat rendering (the
    library's standard training-text convention), `prompt_len` the token
    length of the matching prompt-only rendering (`chat_prompt_text`) --
    an exact prefix of `text` by chat-template determinism (the generation
    prompt is only ever appended to, never rewritten), modulo the
    decode-then-re-encode round trip every binarize_text consumer in this
    library already relies on."""
    from ..axbench.protocol import binarize_text, chat_prompt_text, positive_and_negative_rows

    positive, negative = positive_and_negative_rows(samples, concept)

    def _rows(df):
        out = []
        for row in df.iter_rows(named=True):
            text = binarize_text(tokenizer, row["prompt"], row["response"], suffix_length)
            prompt_text = chat_prompt_text(tokenizer, row["prompt"])
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            question = (row.get("meta") or {}).get("question")
            out.append((text, prompt_len, crop_length(tokenizer, prompt_text, prompt_len, question)))
        return out

    return _rows(positive), _rows(negative)


def cropped_attn_response_avg(model, tokenizer, rows, layer_list):
    """{layer: [N, d] float32 CPU} -- Eq. 7/8's response-position average of
    the prompt-cropped attention-output contribution:
        sys_out_i = o_proj( sum_{j < crop_len} alpha_ij * V_j )
    for each response position i (`alpha` = real post-softmax attention
    weights, already normalized over the full causal history -- the crop
    only restricts which V columns the sum runs over, Eq. 7). `rows`:
    list of (text, prompt_len, crop_len) from `contrast_rows`."""
    import torch

    from .rope import repeat_kv

    num_heads, num_kv_heads, num_kv_groups, head_dim = layer_dims(model)
    out = {l: [] for l in layer_list}

    for text, prompt_len, crop_len in rows:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: t.to(model.device) for k, t in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]
        if prompt_len >= seq_len:
            raise ValueError(f"prompt_len {prompt_len} >= sequence length {seq_len} "
                             "-- no response tokens left to average over")

        captured_v = {}
        cropped_out = {}
        handles = []

        def make_v_hook(l):
            def hook(module, inp, output):
                captured_v[l] = output.detach()
            return hook

        def make_attn_hook(l):
            def hook(module, inp, output):
                attn_weights = output[1] if isinstance(output, (tuple, list)) else None
                if attn_weights is None:
                    raise RuntimeError(
                        "self_attn returned no attention weights -- GCAD extraction needs "
                        "the model loaded with attn_implementation='eager'")
                v_raw = captured_v.pop(l)  # [B, T, num_kv_heads * head_dim]
                bsz, seq_len_v = v_raw.shape[0], v_raw.shape[1]
                v = v_raw.view(bsz, seq_len_v, num_kv_heads, head_dim).transpose(1, 2)
                v = repeat_kv(v, num_kv_groups)  # [B, num_heads, T, head_dim]

                crop_w = attn_weights[:, :, :, :crop_len]
                crop_v = v[:, :, :crop_len, :]
                crop_out = (crop_w @ crop_v).transpose(1, 2).reshape(bsz, -1, num_heads * head_dim)
                cropped_out[l] = module.o_proj(crop_out).detach()
            return hook

        for l in layer_list:
            attn = model.model.layers[l].self_attn
            handles.append(attn.v_proj.register_forward_hook(make_v_hook(l)))
            handles.append(attn.register_forward_hook(make_attn_hook(l)))

        with torch.no_grad():
            model(**inputs, output_attentions=True, output_hidden_states=False, use_cache=False)

        for h in handles:
            h.remove()

        for l in layer_list:
            out[l].append(cropped_out[l][:, prompt_len:, :].mean(dim=1).float().cpu())

    return {l: torch.cat(rows_, dim=0) for l, rows_ in out.items()}


def k_pos_and_avg_demand(model, tokenizer, pos_rows, layer_list):
    """GCAD's gate calibration (paper Eq. 9-10), from the positive rows only (upstream's choice).

    For each layer l in `layer_list`:
      k_pos[l]      = post-RoPE key, averaged over every crop-window token (the instruction span)
      avg_demand[l] = demand d = mean_h(Q . k_pos) / sqrt(head_dim), averaged over response tokens

    At inference the gate compares each new token's demand against avg_demand[l].

    Args:
      pos_rows: [(text, prompt_len, crop_len)] from `contrast_rows`.
    Returns:
      k_pos [Lc, num_kv_heads, head_dim], avg_demand [Lc], both float32 on CPU.

    Plain forward (no eager needed): reads only q_proj / k_proj outputs and the model's rotary_emb.
    """
    import torch

    from .rope import apply_rotary_pos_emb

    num_heads, num_kv_heads, num_kv_groups, head_dim = layer_dims(model)

    # running sums over all positive rows, per layer
    k_sums = {l: None for l in layer_list}   # [num_kv_heads, head_dim], summed over crop tokens
    k_counts = {l: 0 for l in layer_list}    # crop tokens seen
    q_sums = {l: None for l in layer_list}   # [num_heads, head_dim], summed over response tokens
    q_counts = {l: 0 for l in layer_list}    # response tokens seen

    captured_q, captured_k = {}, {}
    handles = []

    def make_q_hook(l):
        def hook(m, inp, out):
            captured_q[l] = out.detach()  # [1, T, num_heads * head_dim], pre-RoPE
        return hook

    def make_k_hook(l):
        def hook(m, inp, out):
            captured_k[l] = out.detach()  # [1, T, num_kv_heads * head_dim], pre-RoPE
        return hook

    for l in layer_list:
        attn = model.model.layers[l].self_attn
        handles.append(attn.q_proj.register_forward_hook(make_q_hook(l)))
        handles.append(attn.k_proj.register_forward_hook(make_k_hook(l)))

    try:
        for text, prompt_len, crop_len in pos_rows:
            # positions: [0, crop_len) instruction | [crop_len, prompt_len) rest of prompt | [prompt_len, T) response
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            inputs = {k: t.to(model.device) for k, t in inputs.items()}
            seq_len = inputs["input_ids"].shape[1]
            if prompt_len >= seq_len:
                raise ValueError(f"prompt_len {prompt_len} >= sequence length {seq_len} "
                                 "-- no response tokens left for the demand average")
            device = inputs["input_ids"].device
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T], unpadded single row

            with torch.no_grad():
                model(**inputs, output_hidden_states=False, output_attentions=False, use_cache=False)

            for l in layer_list:
                q_raw = captured_q.pop(l)
                k_raw = captured_k.pop(l)
                dev = q_raw.device
                q = q_raw.view(1, seq_len, num_heads, head_dim).transpose(1, 2)     # [1, H, T, D]
                k = k_raw.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # [1, Hkv, T, D]
                # RoPE by hand: the hooks see projections before the attention module rotates them
                cos, sin = model.model.rotary_emb(k, position_ids.to(dev))
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
                q_rope = q_rope.float().cpu()
                k_rope = k_rope.float().cpu()

                crop_k_sum = k_rope[0, :, :crop_len, :].sum(dim=1)  # [Hkv, D]: keys of the instruction tokens
                k_sums[l] = crop_k_sum if k_sums[l] is None else k_sums[l] + crop_k_sum
                k_counts[l] += crop_len

                resp_q_sum = q_rope[0, :, prompt_len:, :].sum(dim=1)  # [H, D]: queries of the response tokens
                q_sums[l] = resp_q_sum if q_sums[l] is None else q_sums[l] + resp_q_sum
                q_counts[l] += seq_len - prompt_len
    finally:
        for h in handles:
            h.remove()

    k_pos = torch.stack(
        [k_sums[l] / max(k_counts[l], 1) for l in layer_list], dim=0)  # [Lc, Hkv, D]

    avg_demand = []
    for module_idx, l in enumerate(layer_list):
        if q_sums[l] is None or q_counts[l] == 0:
            avg_demand.append(torch.tensor(0.0))
            continue
        q_mean = q_sums[l] / q_counts[l]  # [H, D] mean response query
        k_per_q = k_pos[module_idx].repeat_interleave(num_kv_groups, dim=0)  # [H, D]: GQA, one kv head per query group
        # demand is linear in Q, so the demand of the mean query == the mean of per-token demands
        d = (q_mean * k_per_q).sum(dim=-1).mean() / (head_dim ** 0.5)  # scalar
        avg_demand.append(d)
    avg_demand = torch.stack(avg_demand, dim=0)  # [Lc]

    return k_pos, avg_demand
