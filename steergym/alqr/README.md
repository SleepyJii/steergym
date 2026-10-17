# steergym.alqr

Port of **A-LQR** ("Local Linearity of LLMs Enables Activation Steering via
Model-Based Linear Optimal Control", arXiv 2604.19018), from the official
repo at commit `c2e0c84`. A-LQR treats the residual stream as a discrete-time
dynamical system -- one time step per decoder layer -- and steers it with a
time-varying LQR controller tracking a contrastive setpoint at every layer.

## The steering object

Per concept, per layer `k = 0..L-1` (`L` = number of decoder layers):

- `v_k` -- the unit contrastive direction (`e_k = mean(pos) - mean(neg)` at
  layer `k`'s INPUT activation, last prompt token; `v_k = e_k / ||e_k||`).
- `mu_k = ||e_k||` -- the contrastive-vector norm; the setpoint target is
  `beta*_k = lambda * mu_k`, `lambda` the method's dose knob (upstream's
  typical range 1-3.5).
- `w_k = K_k @ v_k` -- the Riccati feedback gain row, where `K_k` is the
  time-varying LQR gain from `control.riccati_gains` (upstream's
  `time_varying_lqr_noB_mem_efficient`, isotropic costs `Q = q*I`,
  `R = r*I`, `B = I`, terminal `S_L = q_f*I`).

At inference, upstream's hook computes `u = K_k @ (alpha_k * v_k)` where
`alpha_k = beta*_k - x_k . v_k` is the scalar setpoint error along `v_k`.
Because `K_k` is linear, `K_k @ (alpha_k * v_k) == alpha_k * (K_k @ v_k) ==
alpha_k * w_k` exactly -- not an approximation, ordinary linear algebra. So
the fitted artifact only ever needs the **rank-1** row `w_k [d]`, never the
full `K_k [d, d]` matrix upstream keeps resident at runtime (for an 8B model
at `d ~ 4096`, that's `L * d^2 * 4` bytes ~ 2 GB of gain matrices we never
materialize past the fit).

## Provenance / deltas vs upstream

| delta | justification |
|---|---|
| store `v, w, mu` (one row per layer) instead of the full `[L, d, d]` gain `K` | the rank-1 identity above -- exact, see above |
| jacobians use hidden_states rows `0..L-1` only; row `L` (the last layer's OUTPUT) is never collected | A-LQR's state is the INPUT to each decoder layer; the final layer's output is not a state the method steers |
| explicit 4D additive causal mask (`torch.finfo(dtype).min` upper-triangular) passed into each decoder layer, vs upstream's raw 2D float `attention_mask` passed straight through `tf_block_wrapper` | upstream's mask shape is ambiguous (at best) against a modern transformers decoder layer, which expects a 4D additive mask. **Verified inert for the last-position Jacobian**: the causal mask's last row is fully unmasked, so both maskings agree there (provable, and probed to <=1.5e-7 on tiny random llama/qwen2/gemma2, eager + sdpa) |
| jacobian prompts are LEFT-truncated to their final `jacobian_max_ctx` tokens, vs upstream's right truncation | upstream right-truncates RAW untemplated text; our texts are chat-templated, and right truncation would pin the linearization point to the template header -- on llama-3.1/qwen2.5 instruct templates the header alone exceeds 24 tokens, collapsing all 50 Jacobians to one evaluated at a system-prompt token. The tail is the last-token state family the contrast rows and the controller act on |
| jacobian-prompt subsample is seeded (`random.Random(config.seed).sample(...)`) | upstream's `random.sample` draws from the ambient global RNG; our artifacts must reproduce from `config.seed` alone |
| extraction is right-pad + length-1 index (`steergym.activations`' convention) vs upstream's left-pad-and-take-`[-1]` | equivalent -- both read the same real last token, just addressed from opposite ends of the padded batch |
| extraction collects via `output_hidden_states=True` over one batched forward, vs upstream's forward hooks on the decoder layers during `model.generate(max_new_tokens=1)` | equivalent for rows `0..L-1` -- both read each layer's real input at the prompt's last token; we skip the one-token `generate()` call since the method never needs a generated token during fitting |
| `ALQRContext` clones the hooked output before writing (`h.clone(); h[:, -1, :] = ...`), never upstream's in-place `output[0][..., -1, :] = ...` | `transform.py`'s idiom library-wide; avoids corrupting a tensor another hook or the KV cache may still reference |
| `diagnostics()` (`steered_rows`, `applied_delta_norm_mean/max`) replaces upstream's `SIGNAL_COLLECT` hooks | steergym's cross-method dose/diagnostic convention (`SteerContext.diagnostics()`) |
| no `top_p=0.3` / `repetition_penalty=1.2` baked into generation | upstream's `track_setpoint` hardcodes these into its own `model.generate(...)` call; in steergym, sampling is `GenParams`, a runscript concern, not the steering method's |
| defaults (`q=10, r=10, q_f=1`) are upstream's `LQRSteering.__init__` class defaults, not any published experiment's | **every** released eval script overrides them; the paper body never states per-model values. Per-model settings actually used, for runscripts to reach for: |

| model | task | `[q, r, q_f]` | `lambda` |
|---|---|---|---|
| gemma-2-2b | toxicity | `[0.1, 3, 0.1]` | 1-1.5 |
| gemma-2-2b | TruthfulQA | `[0.1, 1, 0.3]` | 3 |
| llama-3-8b | toxicity & TruthfulQA | `[0.1, 10, 10]` | 2-3.5 |
| qwen2.5-14b | toxicity | `[1, 1, 1]` | 2-2.5 |
| qwen2.5-14b | TruthfulQA | `[0.1, 1, 0.3]` | 3-3.5 |
| (any) | concepts | `[0.1, 1, 0.1]` | 0.5-2.5 |
| (any) | refusal, A-LQR+ (`positions="all"`) | `[0.2, 7, 0.2]` | 1 |

(source: the released eval scripts under `steer/{toxicity,truthfulness,refusal}/`
and `steer/concepts/concept_pipeline.py`, not the paper text.)

`jacobian_prompts=50` / `jacobian_max_ctx=24` are the toxicity/refusal
scripts' settings (their working config for 8-9B models); the concepts
pipeline (`concept_pipeline.py`, `multisteer_pipeline.py`) uses
`max_ctx=32` at the same prompt count -- a runscript-level override, not a
new default. Upstream's linearization pool is the DESIRED-class (`D+`)
prompts in every script (toxicity: `nontoxic_prompts`; refusal: the safe
pool; concepts: the concept sentences; TQA: a debug-state N=1) -- our
"positive texts" pool matches that convention, with count/ctx per-task
knobs.

**Not ported:** `multisteer` (tracking two concepts' setpoints
simultaneously via a second contrastive vector/gain) -- v1 is single-concept,
matching the per-concept `SteerModel.context` convention every other method
in this library follows.

## Op-order note (Riccati recursion)

Upstream's `time_varying_lqr_noB_mem_efficient` computes `K_t = solve(P, F)`
and then, in the very next line, computes `solve(P, F)` a **second time**
for the `S` update rather than reusing `K_t`:

```python
K[t] = th.linalg.solve(P, F)
S[t] = (G - F.transpose(-2, -1) @ th.linalg.solve(P, F)).to('cpu')
```

`control.riccati_gains`/`riccati_gain_rows` keep this exact double solve
(rather than reusing the first `solve(P, F)` result) so our floating-point
trace matches upstream's bit for bit **on the same device and dtype** (the
parity test pins CPU/fp32; a GPU run of either implementation lands on
different BLAS kernels, and the stored `w = K @ v` rounds differently from
upstream's runtime `K @ (alpha v)` in the last bit regardless).
`riccati_gain_rows` additionally never allocates the full `K_t`-history
array -- only `w_t = K_t @ V[t]` survives each step -- but is required
(and tested) to equal `riccati_gains(A, ...)[t] @ V[t]` exactly. The fit
runs the recursion on the subject device: CPU fp32 is ~minutes per layer
at `d >= 4096` (upstream also solves on GPU; its `S`-to-CPU offload is a
memory tactic, numerically a no-op).

## steergym.alqr.spid -- S-PID (the paper's own controller ablation)

S-PID (`steer/PIDsteering.py`'s `track_setpoint` / `hook_setpoint_tracking`,
upstream App E.2) replaces A-LQR's Riccati gain `K` with a per-layer PID over
the SAME setpoint error. No jacobians, no `A`/`K` -- `SPIDModel.fit` is the
DiffMean half of `ALQRModel.fit` only (stop before `averaged_block_jacobians`
and `riccati_gain_rows`), so a fit is ~2 min/concept, not ~70. The paper has
**no equation** for S-PID -- `hook_setpoint_tracking` IS the definition, and
every line below is a citation into it, not a derivation:

```
alpha_k = lambda*mu_k - v_k . x_in          (x_in = layer k's INPUT, last position)
e_k     = alpha_k * v_k
e_sum  += e_k ; if k % integral_reset_every == 0: e_sum = 0     (AFTER accumulating)
u_k     = Kp*e_k + Ki*e_sum + Kd*(e_k - e_prev)
e_prev  = e_k
layer output at position -1 += u_k
```

### Upstream quirks (these ARE the method -- ported as faithful defaults,
knobs to disable, each pinned by a test in `tests/test_spid.py`)

| quirk | upstream cite | our knob |
|---|---|---|
| integral resets every 10th layer (0, 10, 20, ...), **after** accumulating that layer's own error -- a reset layer's `e` is discarded, never reaching the next layer's sum | `PIDsteering.py:172-175` | `integral_reset_every` (default 10; `0`/`None` disables) |
| `e_prev` is never reset within OR between forward passes -- layer 0's D-term at decode step `t` sees the LAST layer's `e` from step `t-1` (cross-decode-step leak) | `PIDsteering.py:177-179` (no reset site anywhere in the hook); reset only happens once per `track_setpoint` call, at `:372` | `d_reset_between_steps` (default `False` = faithful leak; `True` zeroes `e_prev` at each new decode step) |
| state (`e_sum`, `e_prev`, both `[B, d]` fp32 zeros) resets once per `track_setpoint` call == once per `generate()` | `PIDsteering.py:371-372` | our context spans MANY `generate()` calls (one object, many requests) -- see "context state reset seam" below |

The only guard upstream has is `NaN`/`Inf` on `u_t` (`:184-189`), ported
verbatim as a `RuntimeError`.

### Context state reset seam

Upstream's state lifecycle is simple because `track_setpoint` owns exactly
one `generate()` call. Ours doesn't: `SPIDContext` is built once per concept
and may be entered across many separate `generate()` calls. `_state_for`
reconstructs the per-call reset from what's observable at the hook (layer
`k == 0`, input shape `[B, T, d]`): lazy-init zeros on first use; a prefill
(`T > 1`) means a new call started, so state zeros; a batch-size change also
forces a fresh zero-init (unreachable in upstream's own single-call world).
A device change is NEVER a reset trigger: on a sharded model the last
layer's device differs from layer 0's at every decode step, so keying the
reset on it would zero `e_prev` each step (`d_reset_between_steps` without
asking) -- state instead MIGRATES to whichever device the current layer
runs on, at `k == 0` and mid-sweep alike. Known blind spot, impossible on
real text: a generate() whose every prompt is a single token (`T == 1`
prefill, same `B` as the previous call) is indistinguishable from a decode
step, so state would leak across those two calls.

Note the state machinery is INERT at the released llama gains: with
`kd = 0` nothing reads `e_prev`, and `integral_reset_every = 10` wipes
`e_sum` at layer 0 of every forward pass -- the controller is then
memoryless across passes, and only quirk #1 (the within-pass
accumulate-then-reset cadence) shapes behavior. Runs at those gains
exercise quirks #2/#3 structurally, not behaviorally.

### positions, layers, width

`SPIDContext` only supports `positions="generated"` and raises
`ValueError` otherwise -- upstream S-PID has no `positions="all"` /
A-LQR+-style variant (`register_setpoint_tracking_hooks` always hooks
`:198-212`, position `-1` only). It hooks **every** decoder layer, same as
A-LQR; unlike PID-AcT (`steergym.acttransport`), which must width-gate
because its module sets mix sites of different widths, S-PID's sites are
all decoder-layer residual streams of the SAME width `d` -- there is no
analogous gate to add.

### Gains and lambda -- paper-silent, code-pinned (GAP)

The paper states no S-PID gains or lambda-selection rule anywhere. The
library default (`kp=10, ki=10, kd=1`) mirrors `PIDSteering.__init__`'s
constructor defaults (`PIDsteering.py:26-28`) **for API-consistency only**
-- no released script uses them. The released per-model table
(`tqa_evalPID.py:123-159`, `tox_evalPID.py:128`) pins **Llama-3-8B TQA and
toxicity both to `(kp, ki, kd) = (0.1, 0.1, 0.0)`** (`kd=0`: no released
llama cell uses the D term at all). `lambda` is "typically 1-2.5" per the
docstring, with App E.2 noting PID "is incoherent when lambda>=2" on
toxicity; the llama TQA active sweep is `[0.5, 1, 1.5]` (commented-out
alternative `[1.5, 2, 2.5, 3]`). Runscripts should reach for the driver's
own env knobs rather than these library defaults.

### `from_alqr` -- upstream's own artifact-sharing seam

`SPIDModel.from_alqr(alqr_model, **overrides)` builds a fitted `SPIDModel`
by borrowing a fitted `ALQRModel`'s `(v, mu)` rows AS-IS (same tensor
objects, never recomputed) -- this is not a steergym convenience, it's
upstream's OWN design: `PIDSteering.__init__` takes the identical
`contrastive_vecs` upstream also hands `LQRSteering`, and the `A` (jacobian)
argument is simply unused by the PID path. Extraction/subject knobs
(`subject`, `batch_size`, `max_length`, `seed`) carry from the source
config; `kp`/`ki`/`kd`/reset knobs take `overrides` or the library
defaults -- A-LQR's `q`/`r`/`q_f`/jacobian knobs have no SPIDConfig
counterpart and are simply dropped.

### Released-script-state caveat

Same class of caveat as A-LQR's upstream-hazards note: `tqa_evalPID.py`'s
`main` is not a faithful record of the paper's Table 2 S-PID row (its own
`generation()` call is commented out, only the gemma branch is active, and
`num_trials=10`) -- it is read here only for the gain/lambda table, not as
a runnable reference.
