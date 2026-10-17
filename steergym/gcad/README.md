# steergym.gcad

Port of **GCAD** ("Prompt-Activation Duality: Improving Activation Steering
via Attention-Level Interventions", arXiv:2605.10664), from the official
repo [`NihiI-obstat/Gated-Cropped-Attention-Delta`](
https://github.com/NihiI-obstat/Gated-Cropped-Attention-Delta) at commit
`6c879f4`. GCAD steers the attention module's output directly, gated by a
per-token, data-dependent compatibility score, instead of adding a fixed
vector to the residual stream at every position.

## The method (paper Sec. 4, Eq. 6-11)

Per layer `l`, attention output decomposes exactly over source positions
(Eq. 6): `Attn_l(h_t) = sum_{i<=t} alpha_{t,i} V_i W_o`. GCAD:

1. **Cropped (P2, Eq. 7)**: restrict the sum to a fixed window `S` of source
   positions (upstream: the system-prompt tokens) -- `Attn_S(h_t) =
   sum_{i in S} alpha_{t,i} V_i W_o`.
2. **Attention-delta (P1, Eq. 8)**: `v[l] = E_D+[Attn_S(h_pos)] -
   E_D-[Attn_S(h_neg)]`, averaged over response-token positions, each side's
   expectation over its own contrastive pool.
3. **Gated (P3, Eq. 9-10)**: at inference, a per-token demand score
   `d_i[l] = mean_h(Q_i[l,h] . K_pos[l,h//ng]) / sqrt(head_dim)` (Eq. 9,
   post-RoPE Q against `K_pos` = post-RoPE K averaged over `S` on the
   positive pool), centered against `avg_demand[l]` (the same demand
   averaged over response positions, positive pool) and squashed through a
   centered sigmoid: `c_i[l] = 2 * base_coef * sigmoid(scale * (d_i[l] -
   avg_demand[l]))` (Eq. 10).
4. **Inference update (Eq. 11)**: `attn_out_steered = attn_out + c_i[l] *
   v[l]`, applied to the attention SUBMODULE's output -- before it feeds
   this layer's own residual-add + MLP, and before the NEXT layer computes
   its K/V from it.

## Family chosen: a new multi-layer hook context (like A-LQR/S-PID)

Neither `RowVectorModel` (single-site residual-stream row addition) nor
`TransformSteerContext` (`h -> f(h)` at one hook point per layer, the
decoder layer's own output) fits: GCAD (a) injects at the **attention
submodule's** output, not the decoder layer's, and (b) needs a per-token
gate computed from **live Q** (captured via a `q_proj` forward hook,
RoPE'd by hand from a `position_embeddings` pre-hook) against **fitted K**
(`k_pos`) -- three hooks per layer, and two calibration tensors
(`k_pos`, `avg_demand`) per layer beyond the row `v`. This is the same
category of chassis gap A-LQR/S-PID hit (`steergym/alqr/README.md`):
`GCADContext` is a new `SteerContext`, following their precedent rather
than forcing the method into either existing family.

**Scope limit** (shared with upstream): assumes the standard HF
`q_proj`/`k_proj`/`v_proj`/`o_proj` + GQA attention module layout
(Llama/Qwen2/Mistral/Gemma2/...) -- exactly the models the paper tests
(Qwen2.5, Llama-3.1). Fused-qkv architectures (GPT-2/Falcon-style) are out
of scope, same as every other submodule-hooking method in this library.

## Deltas vs upstream

| delta | justification |
|---|---|
| **crop window S = the rendered prompt up to `meta["question"]`** (template prefix + folded instruction); whole prompt when a row has no question suffix | upstream crops to the system turn (`"<|im_end|>\n"` search in a persona-vectors chat string). steergym folds the instruction into `prompt` ahead of the question (personavec `fold`), so the question's first token is the boundary -- `extract.crop_length`, via the tokenizer's offset mapping. Cropping the question too would add question-token V (shared by both pools, so mostly cancelling) to `v` and, worse, average question-token K into `k_pos`, diluting the gate. Parity leg runs with a non-empty user turn and matches upstream exactly. Benchmarks with no question split (AxBench) crop to the whole prompt |
| `prompt_len` from `axbench.protocol.chat_prompt_text` + re-tokenization, vs upstream's raw `tokenizer.encode(prompt, add_special_tokens=False)` on a pre-templated string | upstream's "prompt" field is already a rendered chat-template string (baked in at dataset-prep time, outside this method); steergym's canonical `prompt` is raw text needing `binarize_text`/`chat_prompt_text` rendering, the library's standard convention (A-LQR's `_contrast_texts`, odesteer). Relies on the same decode-then-re-encode round trip every `binarize_text` consumer in this library already assumes |
| RoPE application + GQA repeat implemented locally (`rope.py`) instead of importing `transformers.models.qwen2.modeling_qwen2.{apply_rotary_pos_emb,repeat_kv}` | both functions are copy-identical across every rotate-half+GQA HF decoder; importing one architecture's module path ties the port to that path existing for no behavioral gain. Verified bit-exact against the real package's own copies (parity leg) |
| dose vocabulary `strength` (`context(concept, strength=1.0, ...)`), `base_coef = strength * config.base_coef` | library convention (A-LQR's `lmbda`, odesteer's `T`) -- upstream's own `base_coef`/`inverse` constructor args are config, not a per-call dose knob |
| no `inverse` flag | upstream's `inverse=True` flips the gate's sign (steer away from the trait); omitted in v1 -- `strength` can already be negative for the same effect, since `coef` is linear in `base_coef` |
| default `positions="all"` | upstream's `MultiLayerGatedSteerer` steers every position (prompt + generated); `"generated"` (last position per forward) is available as an ablation |
| unbatched fit-time extraction (one forward per row) | matches upstream's own loop exactly (`get_prompt_attn_delta_p_and_r`/`compute_k_pos_and_avg_demand` never batch either) -- fit-time cost only, not a generation hot path |
| `avg_demand`/`k_pos` computed from the positive pool only | upstream's own choice (`save_trait_prompt_k_pos` reads only `pos_prompts`/`pos_responses`) -- carried through unchanged |

## Hyperparameters (paper Appendix A)

| model | `layers` | `base_coef` (c_base) | `scale` (s) |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 9-19 (11 layers) | 3.5 | 1.5 |
| Llama-3.1-8B-Instruct | 8-20 (13 layers) | 2.0 | 1.5 |

`GCADConfig.layers=None` (fit over every decoder layer) and
`base_coef=2.0`/`scale=1.5` (the Llama row) are the library defaults for
API-consistency, matching the A-LQR precedent of "defaults are *a* paper
cell's values, not a universal recommendation" -- runscripts should pass
the per-model `layers`/`base_coef` from the table above.

## Fit requirements

`fit()` needs the bound model loaded with `attn_implementation="eager"`:
SDPA/FlashAttn never populate `attn_weights` even when
`output_attentions=True` is passed (upstream's own comment,
`core/generate_vec.py:252-253`) -- `cropped_attn_response_avg` raises a
clear `RuntimeError` rather than silently extracting a null vector if this
is missed.

## Tests

`tests/test_gcad.py`: RoPE/repeat_kv math oracles (+ parity vs the real
`transformers` package's own Qwen2/Llama copies when installed), the Eq.
9/10 gate formula on hand-built Q/K, the Eq. 6/7 exact-decomposition oracle
on a real tiny HF model (cropping to the full sequence must reproduce the
model's own uncropped `self_attn` output bit-for-bit -- the causal mask
already zeros everything the crop would otherwise include), degenerate
limits (`strength=0` is an exact identity; `positions="generated"` vs
`"all"`), closed-loop KV-cache generate equivalence (batched vs. alone),
fit determinism + save/load round trip on a real tiny HF model, and a
`GCAD_DIR`-gated leg replaying upstream's own `gated_steerer.py`/
`generate_vec_prompt_attn_k.py` functions against ours on identical
tensors (skips cleanly if the clone or `transformers` is absent). GPU
untested.
