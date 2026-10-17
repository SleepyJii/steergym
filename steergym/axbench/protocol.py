import polars as pl

from ..concept import Concept

# ---------------------------------------------------------------- template/prefix/suffix helpers
# The latent max-act
# pass and the native fit_* loops below both need the chat-template prefix/suffix machinery.
# released_dicts.py now imports these instead of carrying its own copies.


def template_ids(tokenizer, messages, add_generation_prompt: bool = False) -> list[int]:
    """Flat token-id list from apply_chat_template across transformers versions:
    the 4.45 era returned list[int]; arena 5.x returns a BatchEncoding."""
    out = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=add_generation_prompt)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def prefix_suffix_lengths(tokenizer) -> tuple[int, int]:
    """axbench model_utils.get_prefix_length/get_suffix_length, common_prefix=None
    branch: tokenize two different single-token user turns through the chat
    template (no add_generation_prompt); prefix = first divergence index,
    suffix = count of common trailing ids."""
    tokens_a = template_ids(tokenizer, [{"role": "user", "content": "1"}])
    tokens_b = template_ids(tokenizer, [{"role": "user", "content": "2"}])
    prefix_length = 0
    for i, (ta, tb) in enumerate(zip(tokens_a, tokens_b)):
        if ta != tb:
            prefix_length = i
            break
    suffix_length = 0
    for i, (ta, tb) in enumerate(zip(reversed(tokens_a), reversed(tokens_b))):
        if ta != tb:
            suffix_length = i
            break
    if prefix_length == 0 or suffix_length == 0:
        raise RuntimeError(f"degenerate chat template: prefix_length={prefix_length} "
                           f"suffix_length={suffix_length} ([1:-0] would empty every latent text)")
    return prefix_length, suffix_length


def latent_texts(latent_df: pl.DataFrame, tokenizer, suffix_length: int) -> list[str]:
    """axbench inference.py prepare_df (non-Llama branch): BOS stripped
    ([1:]), trailing suffix_length ids stripped ([:-suffix_length]), decoded.
    latent_df is canonical samples schema (prompt/response), e.g.
    Concept500Benchmark.latent_samples()."""
    texts = []
    for row in latent_df.iter_rows(named=True):
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        tokens = template_ids(tokenizer, messages)[1:-suffix_length]
        texts.append(tokenizer.decode(tokens))
    return texts


def latent_max_acts(generator, steer_model, latent_df: pl.DataFrame, prefix_length: int,
                    suffix_length: int, batch_size: int, relu: bool, layer: int,
                    bias: float | None = None) -> dict[int, float]:
    """axbench lsreft.py predict_latent mechanics, replicated against our own
    raw row (not the trained ax module): per-token score = relu(h.v) [LsReFT]
    or raw h.v [DiffMean] at positions [prefix_length:], trimmed per example
    to attention_mask.sum() - prefix_length (right padding). Per-row max
    rounded to 3dp (axbench rounds); per-concept max over rows; <=0 -> 50.0.
    `bias` (PreferenceVector's trained bias, added before the relu: relu(h.v +
    b)) is a single scalar applied to every concept in latent_df -- callers
    with per-concept biases (native-refits PV artifacts) call this once per
    concept. Default None leaves scores unchanged, so LsReFT/DiffMean callers
    (bias always 0 for them) are byte-for-byte unaffected."""
    import torch

    tokenizer = generator.tokenizer
    tokenizer.padding_side = "right"
    layer_module = generator.model.model.layers[layer]
    max_acts = {}

    for cid in sorted(latent_df.get_column("concept_id").unique().to_list()):
        cid = int(cid)
        concept_rows = latent_df.filter(pl.col("concept_id") == cid)
        v = steer_model.row_for(Concept(id=cid, name="", description=""))
        texts = latent_texts(concept_rows, tokenizer, suffix_length)

        row_maxes = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", add_special_tokens=True,
                            padding=True, truncation=True)
            enc = {k: t.to(generator.model.device) for k, t in enc.items()}

            captured = {}

            def hook(mod, args, output, captured=captured):
                captured["h"] = output[0] if isinstance(output, tuple) else output

            handle = layer_module.register_forward_hook(hook)
            with torch.no_grad():
                generator.model(**enc)
            handle.remove()

            h = captured["h"]  # [B, T, d]
            seq_lens = enc["attention_mask"].sum(dim=1) - prefix_length
            vv = v.to(device=h.device, dtype=h.dtype)
            scores = h[:, prefix_length:] @ vv  # [B, T - prefix_length]
            if bias is not None:
                scores = scores + bias
            if relu:
                scores = torch.relu(scores)

            for b in range(scores.shape[0]):
                n = max(int(seq_lens[b].item()), 0)
                row_scores = [round(x, 3) for x in scores[b, :n].tolist()]
                row_maxes.append(max(row_scores) if row_scores else 0.0)

        concept_max = max(row_maxes) if row_maxes else 0.0
        max_acts[cid] = concept_max if concept_max > 0 else 50.0

    return max_acts


# ---------------------------------------------------------------- axbench data-prep helpers
def positive_and_negative_rows(samples: pl.DataFrame, concept: Concept) -> tuple[pl.DataFrame, pl.DataFrame]:
    """axbench train.py's prepare_df filtering: positives = this concept's
    output_concept + category=='positive' rows; negatives = the global
    concept_id==-1 pool, genre-matched. `samples` is canonical schema (e.g.
    Concept500Benchmark.train_samples()): the native axbench columns ride
    verbatim in `meta`. Genre is read off the positive rows' own
    meta.concept_genre rather than a separate metadata lookup (the positive
    rows for one concept share a single genre) -- see FIDELITY NOTES."""
    meta = pl.col("meta")
    category = meta.struct.field("category")
    positive = samples.filter(
        (meta.struct.field("output_concept") == concept.name) & (category == "positive"))
    if positive.height == 0:
        negative = samples.filter(
            (pl.col("concept_id") == -1) & (category == "negative")).head(0)
        return positive, negative
    genre = positive.get_column("meta").struct.field("concept_genre")[0]
    negative = samples.filter(
        (pl.col("concept_id") == -1) & (category == "negative")
        & (meta.struct.field("concept_genre") == genre))
    return positive, negative


def chat_prompt_text(tokenizer, text: str) -> str:
    """axbench prepare_df's non-binarize chat templating: no system prompt
    (matches GeneratorLM.chat_prompt), BOS stripped then re-added at
    tokenization time."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    bos = tokenizer.bos_token
    if bos and rendered.startswith(bos):
        rendered = rendered[len(bos):]
    return rendered


def binarize_text(tokenizer, input_text: str, output_text: str, suffix_length: int) -> str:
    """axbench prepare_df's binarize=True chat templating (non-system-prompt
    branch): [user, assistant] through apply_chat_template with
    add_generation_prompt=True, BOS + trailing suffix_length ids stripped,
    decoded."""
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    ids = template_ids(tokenizer, messages, add_generation_prompt=True)[1:-suffix_length]
    return tokenizer.decode(ids)
