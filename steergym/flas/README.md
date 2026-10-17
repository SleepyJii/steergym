# steergym.flas

Port of **FLAS** ("Beyond Steering Vector: Flow-based Activation Steering",
arXiv 2605.05892, NeurIPS 2026), from the official repo at
`github.com/flas-ai/FLAS` commit `dc8b84c`. `CachedFLASSteerModel` is
**inference-only**: it loads one of the authors' published FlowBlock
checkpoints from the Hugging Face Hub and applies it; there is no fit/
training path (the base `SteerModel.fit`'s "does not train" error is exactly
right here and is not overridden).

## The method

FLAS learns a concept-conditioned velocity field and Euler-integrates it
over "flowtime" `T` on a frozen subject model's residual stream at one
layer:

```
h' = phi_T(h) = h + integral_0^T v_theta(phi_t(h), t, c) dt
             ~= Euler, n_steps steps: h_{k+1} = h_k + dt * v(h_k, t_k, c),  dt = T / n_steps, t_k = k*dt
```

`v_theta` is a **FlowBlock**: TimeEmbed -> CrossAttn(Q=activations,
KV=encoded concept text) -> causal SelfAttn (over the activation sequence's
own positions) -> gated MLP, each wrapped in a residual + learnable
per-channel gate (`flow_model.py`'s `FlowBlock`/`FlowFunction`). The concept text
is encoded by a **frozen** 2-layer slice of the subject model itself
(embeddings + first 2 decoder layers + final norm -- `ConceptEncoder`), not
a separate text encoder. `T` is the dose knob; `n_steps` (the paper uses 3)
is the Euler step count, both method-owned per the house no-universal-dose
rule.

`FlowBlock`/`FlowFunction`/`ConceptEncoder`/the model-family `ArchSpec`
registry are ported essentially verbatim from upstream's `src/flas/model.py`
(gemma2/qwen3/llama/gemma3 families). Dropped entirely: the training-time
constructors (`build_flow_model`, `ConceptEncoder`'s from-scratch
`__init__` that loads a second full base-model copy, `build_chat_prompt`) --
inference always shares modules with the already-bound subject
(`ConceptEncoder.from_base_model`), and prompt chat-templating is already
steergym's job (`GeneratorLM.chat_prompt`), not this method's.

## Registry

`FLAS_CHECKPOINTS` (`models.py`) is the only way in -- `CachedFLASSteerModel`
accepts no ad-hoc checkpoint path or architecture override, only
`from_pretrained(hf_repo)` against this table:

| `hf_repo` | subject | layer | `model_type` | hidden | self-attn | prompt format |
|---|---|---|---|---|---|---|
| `flas-ai/flas-gemma-2-2b-it` | google/gemma-2-2b-it | 20 | gemma2 | 2304 | on | chat |
| `flas-ai/flas-gemma-2-9b-it` | google/gemma-2-9b-it | 20 | gemma2 | 3584 | on | chat |
| `flas-ai/flas-llama-3.1-8b-instruct` | meta-llama/Llama-3.1-8B-Instruct | 20 | llama | 4096 | off | chat |
| `flas-ai/flas-llama-3.1-8b` | meta-llama/Llama-3.1-8B | 20 | llama | 4096 | off | alpaca |
| `flas-ai/flas-qwen3-8b` | Qwen/Qwen3-8B | 20 | qwen3 | 4096 | off | chat |
| `flas-ai/flas-qwen3-8b-base` | Qwen/Qwen3-8B-Base | 20 | qwen3 | 4096 | off | alpaca |
| `flas-ai/flas-gemma-3-4b-it` | google/gemma-3-4b-it | 20 | gemma3_text | 2560 | off | chat |
| `flas-ai/flas-gemma-3-4b-pt` | google/gemma-3-4b-pt | 20 | gemma3_text | 2560 | off | alpaca |

Every row's `model_id`/`layer`/`num_blocks` (always 1)/`n_steps` (always 3)/
`disable_self_attn`/`prompt_format` is read verbatim from that repo's own
`config.json` on the Hub (checked 2026-10-12); `model_type`/`hidden_size`
are read from the checkpoint's own `weight.safetensors` header
(`blocks.0.cross_attn.q_proj.weight` shape -- `[num_heads*head_dim,
hidden_size]`) rather than from the (gated, auth-required)
`google/gemma-*` / `meta-llama/*` repos' configs, so building the registry
needs no gated access. `prompt_format` is informational only: base-model
(`alpaca`/`pt`/`-base`) prompt formatting is a runscript/benchmark concern
elsewhere in the library (the `AxbenchPromptSteerModel`/alpaca-eval
precedent), not something this method applies itself.

Mismatch refusal (`bind()`, before any download): the bound generator's
`get_text_config(model.config).model_type` and `training_helpers.
hidden_size(model, layer_module)` must match the registry entry exactly,
or `bind()` raises `FLASCheckpointError` immediately -- no partial load,
no silent shape mismatch deeper in `load_state_dict`.

Weights are fetched lazily (`huggingface_hub.hf_hub_download`) inside
`context()` -> `_ensure_loaded()`, never at import time or at
`from_pretrained()` (which only builds config from the registry dict).
There is no `save()`/`load()` -- the published checkpoint IS the
persistence; `from_pretrained(hf_repo)` reconstructs it deterministically
every time.

## Concept conditioning

`concept.description or concept.name` (the `fill_expression_prompt`
fallback convention, `steergym.evals.judges`) is plain-tokenized (no chat template --
upstream's own `encode_concept` does a bare `tokenizer(text, ...)`, matching
e.g. `"Talk like a pirate"` for AxBench concept500, whose `description=""`
always) and run through the frozen `ConceptEncoder` once per `context()`
call to get `concept_hidden [1, kv_len, d]` / `concept_mask [1, kv_len]`.

## Decoding-cache design

The FlowBlock's causal self-attention runs over the activation sequence's
own positions, at **every** position of the current forward call -- prompt
tokens at prefill, each new token at decode (FLAS has no "generated" vs
"all" positions choice the row/transform method families have; it is
baked into the architecture, upstream's `generate.py` `_hook_fn`).

Upstream's own choice, ported verbatim in `CachedFLASContext._step`:
**incremental KV caching, never full recompute.** One `DynamicCache` per
FlowBlock per **Euler step** (`_sa_caches[k]`, `k = 0..n_steps-1`) --
because each step's self-attention runs on a *different* intermediate
activation state (`h` after `k` prior Euler updates), they cannot share one
cache. At prefill the caches are built fresh over the whole prompt in one
batched call; at each decode step they extend by one position, exactly like
the subject model's own KV cache. A full recompute alternative would
re-run every prior position's entire `n_steps`-deep Euler trajectory from
scratch on every new token -- `O(n^2)` in sequence length instead of
`O(n)`, and is NOT what upstream does. `test_cached_flas.py` section 6
proves the two are numerically equivalent (incremental result vs. a
from-scratch recompute of the whole accumulated prefix at every Euler
level, float32 tolerance `1e-5`) rather than assuming it.

Prefill-vs-decode has no reliable signal from a plain forward hook (no
access to the generation loop), so it is inferred the way
`steergym.alqr.spid.SPIDContext` already does: `q_len > 1`, or no cache
yet, or a batch-size change, each resets `_sa_caches`/`_past_len`. Same
accepted blind spot: a `generate()` call whose every prompt is a single
token is indistinguishable from a decode step (never reachable on real
text).

Padding: `CachedFLASContext.__enter__` registers its hook with
`with_kwargs=True` to read the subject model's **own** `attention_mask`/
`position_ids` kwargs for the hooked decoder layer, rather than
recomputing them independently the way upstream's `FlasGenerator` does
(which owns the whole generation loop and threads `attention_mask`/
`position_ids` through by hand) -- a bare `SteerContext` forward hook has
no generation-loop hook, so this is the correct seam. `_padding_mask_from_
attn_mask` derives FlowBlock's `[B, kv_len]` 0/1 padding mask from the
subject's processed mask: the **last** query row is causally eligible for
every kv position, so wherever *it* is masked is exactly padding --
exact for a `full_attention` steering layer, or any layer whose sliding
window exceeds the sequence length (true for AxBench-length prompts well
under gemma2's 4096-token window; same "verified inert" class of delta as
`steergym.alqr`'s mask note). `test_cached_flas.py` section 5 proves
padding invariance directly: a sequence run alone vs. as the longer row of
a left-padded batch of 2 produces identical generated tokens and matching
intermediate hidden states (not merely "looks right").

## Deltas vs upstream

| delta | justification |
|---|---|
| `ConceptEncoder.__init__` takes already-resolved modules (`embed_tokens`, `layers`, `norm`, `rotary_emb`, ...) directly; `from_base_model` is the only public constructor | the training-time constructor (loading a second full base-model copy) is out of scope for an inference-only port |
| steering application is a `register_forward_hook(..., with_kwargs=True)` reading the subject's own `attention_mask`/`position_ids`, not a generation-loop-owning driver class (`FlasGenerator`) | `SteerContext` is a hook attached externally by `steer_sweep`/the benchmark harness, which owns `generate()`; there is no generation loop for `CachedFLASContext` to own |
| no `_iv` intervention-ablation knob (upstream's "apply only these Euler steps" / "scale the net displacement" analysis hooks) | that is an ablation/analysis tool, not part of applying the method; a future `SteerAnalysis` instance (the `TrainKLPathAnalysis` precedent) is the right home if needed |
| `concept_max_len` default 64, `n_steps` default from the registry (3) | upstream's own `encode_concept`/CLI defaults |
| dtype: the flow module's own parameter dtype drives computation (bf16 as published, never forced to fp32) | matches upstream exactly (`FlasGenerator._flow_dtype`); unlike the odesteer/acttransport transform families, which ARE fp32-native fits |

## Tests

`steergym/tests/test_cached_flas.py`, plain asserts, CPU, no network by
default (29 checks; two gated legs skip cleanly without their resource):
registry validation + unknown-repo refusal; `bind()` mismatch refusal
(wrong `model_type`, wrong `hidden_size`, matching case accepted);
`_padding_mask_from_attn_mask` unit checks; Euler integration vs. a
hand-computed toy field (`v = alpha*h`, closed form `h_0*(1+dt*alpha)
**n_steps`) plus `T=0` exact identity; the same `T=0` identity and
nonzero-dose diagnostics on a real tiny `FlowFunction`+`ConceptEncoder`;
steered-positions padding invariance; cached-vs-recompute decoding
equality; dtype round-trip (float64 flow weights under a float32 stream);
an `FLAS_DIR`-gated upstream source-parity leg (same-seed module order,
bit-exact velocity field, bit-exact integrated Euler output vs. a
hand-rolled loop using the real upstream `FlowFunction` class); a
real-checkpoint leg that downloads `flas-ai/flas-gemma-2-2b-it`'s 195MB
`.safetensors` (skips cleanly offline/no cache) and `load_state_dict`s it
`strict=True` into our `FlowFunction` -- **verified 2026-10-12: zero
missing/unexpected keys**, confirming the port's module/parameter naming
matches the published checkpoint exactly; `Gemma2Config()`'s own defaults
are google/gemma-2-2b-it's architecture (cross-checked against this
checkpoint's tensor shapes), so this leg needs no access to the gated
base-model repo.

**GPU-untested.** No cell has run the ported FlowBlock steering a real
generation end to end on GPU; the real-checkpoint leg only exercises
`load_state_dict`. Suggested first cell: `flas-ai/flas-gemma-2-2b-it` on
the AxBench Concept500 setup (`Concept500Benchmark`, gemma-2-2b-it, layer
20) at the paper's `T=2`, `n_steps=3`.

## Subject dtype

Upstream's `load_generator` loads the subject in **bf16**, and the concept
encoder (`ConceptEncoder.from_base_model`, the subject's own embeddings +
first 2 layers) inherits that dtype; the published FlowBlocks are bf16.
Here the concept encoder inherits whatever dtype the bound subject has, so
a fp32 subject (the steergym default) encodes concepts and runs layers
0-19 in fp32 -- a precision delta, not a semantic one. Benchmark runs
meant to be comparable with the paper should load the subject in bf16.

