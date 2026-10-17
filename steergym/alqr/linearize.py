"""A-LQR's per-layer Jacobians (upstream steer/lqr_utils.linearize +
data_handling.collect_jacobians). Per unpadded sequence, per layer k:

    A_k = d(block_k output at last position) / d(block_k input, last position)

evaluated at the layer's real input (hidden_states[k] from one no-grad
forward), averaged over prompts. Torch imported lazily: `import steergym`
must stay usable in a bare venv."""


def block_jacobians(model, input_ids):
    """input_ids: [1, L_seq] long tensor on model.device, one unpadded
    sequence. Returns A [L, d, d] float32 CPU."""
    import torch

    layers = model.model.layers
    L = len(layers)
    with torch.no_grad():
        attention_mask = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
        hidden_states = out.hidden_states  # [0..L-1] are each layer's input

    device = input_ids.device
    seq_len = input_ids.shape[1]
    dtype = hidden_states[0].dtype
    d = hidden_states[0].shape[-1]

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    # explicit 4D additive causal mask -- upstream passes its raw 2D float
    # mask into the block. Verified inert for the LAST-row jacobian (the
    # causal mask's last row is fully unmasked): probed <=1.5e-7 on tiny
    # llama/qwen2/gemma2, eager+sdpa (README delta)
    causal_mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(dtype).min,
                             device=device, dtype=dtype).triu(1)

    A = torch.zeros(L, d, d, dtype=torch.float32)
    for k in range(L):
        x = hidden_states[k].detach().clone().requires_grad_(True)
        position_embeddings = model.model.rotary_emb(x, position_ids)

        def f(x_full, layer=layers[k], pe=position_embeddings):
            out_k = layer(x_full, attention_mask=causal_mask,
                         position_ids=position_ids, position_embeddings=pe)
            h = out_k[0] if isinstance(out_k, tuple) else out_k
            return h[0, -1, :]

        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(f, x, vectorize=True)
        # jac: [d (out), 1, L_seq, d (in)] -- slice the last input position
        A[k] = jac[:, 0, -1, :].detach().float().cpu()
    return A


def averaged_block_jacobians(model, tokenizer, texts, max_ctx, log=lambda msg: None):
    """A [L, d, d] float32 CPU: block_jacobians averaged over `texts`
    (upstream's A_sum / N, num_tokens=1 -- one jacobian per prompt),
    each text LEFT-truncated to its final max_ctx tokens."""
    import time

    import torch

    A_sum = None
    for i, text in enumerate(texts):
        t0 = time.time()
        # keep the LAST max_ctx tokens: our texts are chat-templated, and
        # upstream-style right truncation would pin the linearization point
        # to the template header (identical across prompts on llama/qwen
        # templates); the tail is the last-token state family the
        # controller acts on (README delta)
        ids = tokenizer(text)["input_ids"][-max_ctx:]
        input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        A = block_jacobians(model, input_ids)
        A_sum = A if A_sum is None else A_sum + A
        log(f"jacobian prompt {i + 1}/{len(texts)}: {time.time() - t0:.1f}s")
    return A_sum / len(texts)
