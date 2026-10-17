# steergym.odesteer

Ports from [odesteer](https://github.com/ZhaoHongjue/odesteer) — the official
implementation of *ODESteer: A Unified ODE-Based Steering Framework for LLM
Alignment* (ICLR 2026) — which bundles a family of modern activation-steering
methods behind one ~350-line interface. This package brings that family into
steergym so every method rides the same benchmark path as
LsReFT/DiffMean.

## Stage 1 (this package today): the contrastive vector pack

odesteer's `VecSteer` family — methods whose artifact is a single steering
vector. They are `RowVectorModel`s: steering and persistence are the
inherited generic row path (addition at all positions, dose =
factor × max_act, torch checkpoint save/load), so their numbers land on the
same axes as every existing row model. They deliberately do NOT subclass
`AxbenchRowModel` — that adds only axbench dump/HF-dictionary interop, and
none of these methods has a released axbench artifact to load.

| class | weight stem | reduction (over last-token pos/neg activations) |
|---|---|---|
| `CAAModel` | `CAA` | mean(pos) − mean(neg) (odesteer `_caa.py`; Rimsky et al., ACL 2024) |
| `RepEModel` | `RepE` | PCA-1 of positionally-paired diffs, positive-ward oriented (odesteer `_repe.py`; Zou et al., 2023) |
| `ITIProbeModel` | `ITIProbe` | L2-regularized logistic-probe weight vector (odesteer `_iti.py`) |

All rows are unit-normalized at fit time (odesteer normalizes ITI explicitly
and sweeps `T` for the rest; our dosing convention needs unit rows
throughout).

### Fit input

`positive_and_negative_rows` — the same per-concept pos/neg view DiffMean
fits on (concept500: concept-positive rows vs the genre-matched shared
negative pool). Texts go through axbench `binarize_text` (chat template,
BOS + trailing suffix stripped), so each text ends at the response's last
content token and "last token" is the response's own EOS position —
odesteer's `extract_prompt_eos_activations` site, house formatting.
(`LastTokenContrastConfig` is our name — odesteer calls the site "EOS";
"last token" is what it is.)

The extraction itself (`last_token_activations`, on
`batched_layer_activations`) lives in `steergym.activations`, the shared
machinery both this pack and DiffMean's pooled fit run on — it is general,
not odesteer's or axbench's.

### Deltas vs odesteer (all deliberate)

| delta | why |
|---|---|
| extraction texts are chat-templated via `binarize_text`, not raw `Q: {q}\nA: {a}` | our subject is an -it model; matches every other steergym fit |
| pos/neg are concept500's unpaired sets, not same-prompt preference completions | the benchmark's data model; RepE's pairing was positional truncation in odesteer too, so the reduction is unchanged |
| steering = row addition at **all** positions, dose = factor × max_act | the axbench convention every baseline curve uses; odesteer transforms only each newly generated position with strength `T` |
| `RepE` PCA sign is oriented positive-ward (non-negative dot with the mean diff) | sklearn's `svd_flip` sign is a numerical accident; an anti-concept steering row is a latent bug we don't port |
| fits are torch-native (SVD, full-batch LBFGS), no sklearn | zero new dependencies; both objectives are convex/closed-form so the optimum is solver-independent (the logistic objective matches sklearn's default: `softplus` sum + `‖w‖²/2C`, intercept unpenalized) |
| `ITIProbe`, not `ITI` | odesteer's version is a single-layer resid probe — not the paper's per-head top-K + sigma intervention. The honest name keeps `ITI` free for a faithful port |

### CAA vs DiffMean is an extraction-site ablation

`caa_row` is DiffMean's reduction. The entire difference between the two
models is *where* the activation is read: one vector per text at its last
content token (CAA) vs pooled over every non-prefix token (DiffMean). Their
benchmark comparison is therefore a clean ablation of extraction site, for
free.

## Stage 2 (this package now): the transform family

odesteer's state-dependent methods do not reduce to a row + dose — they map
`h → f(h)` inside the hook. The chassis is `steergym/transform.py`:
`TransformSteerContext` applies a per-concept transform at `positions =
"generated"` (odesteer's convention: position −1 of every forward — the
last prompt token at prefill, then each newly generated token under
KV-cache decode; `GeneratorLM` left-pads, so −1 is never a pad) or `"all"`
(the axbench site), and `diagnostics()` reports the
live-accumulated applied `‖f(h)−h‖` — transforms can't precompute dose.
`TransformSteerModel` holds per-concept fitted transform Modules and
persists them as one torch checkpoint. `RowVectorModel.context` gained the
same `positions` knob, so the stage-1 vector pack can also be applied
odesteer-natively: RepE/ITI upstream are unit vectors swept over `T`
(`strength=T` directly); CAA is the one method odesteer runs
*unnormalized* at T=1, so fits now record `raw_norms` and
`strength=raw_norms[i]` reconstructs their raw mean-diff exactly.

The methods (pure fit/steer transforms in `transforms.py`; per-concept
model wrappers + configs in `transform_models.py`; hyperparameter defaults
= the paper's confs):

| class | weight stem | transform (fit over the same last-token pos/neg view as stage 1) |
|---|---|---|
| `LinActModel` | `LinAcT` | coordinate-wise linear optimal-transport `w⊙h + b` (odesteer `_lin_act.py`, verbatim in torch: truncate-pair, mean-center, per-coordinate sort, `w = Σts/Σ(s²+ε)`) |
| `MiMiCModel` | `MiMiC` | logistic gate (pos=1) + Gaussian optimal-transport map neg→pos applied to rows predicted negative (odesteer `_mimic.py`); OT closed form matches POT's `empirical_bures_wasserstein_mapping` with `reg=1e-2` on both covariances |
| `StepODESteerModel` | `StepODESteer` | `h + T·v(h)`, `v` = unit-normalized gradient of a kernel-logistic density-ratio classifier over `NormedPolyCntSketch` features (`sketch.py`, ported line-faithful from `_poly_cnt_sketch.py`; paper conf: degree 2, 8000 components, γ=0.1, c₀=1.0) |
| `ODESteerModel` | `ODESteer` | fixed-step integration of the same field for time `T` (paper: euler, 10 steps). Native solver replaces the `torchdiffeq` dependency — euler/midpoint/rk4 (torchdiffeq's 3/8-rule variant), parity-tested against `odeint` |

Every transform exposes odesteer's `.steer(X, T)` convention, so fitted
objects inject directly into the upstream harness as `model.steer_model`
for oracle comparison. `T` is odesteer's vocabulary — integration
time / strength; `LinAcT` and `MiMiC` ignore it, exactly as upstream. Per
paper appendix C.3, published numbers use the Table-5 per-model `T` ranges
(best-of-sweep), ODESteer included.

### Stage-2 deltas vs odesteer (all deliberate)

| delta | why |
|---|---|
| torch-native fits: LBFGS logistic (shared `logistic_fit`), eigh matrix-sqrt | no sklearn/POT/torchdiffeq deps; convex/closed-form objectives are solver-independent (source-parity in the suite) |
| count-sketch hash draw seeded from `config.seed`; realized hash buffers saved in the artifact | upstream draws from the ambient global RNG — artifacts must reproduce from config alone, and a loaded model must steer with exactly the features it was fitted with |
| fit input = concept500's chat-templated unpaired pos/neg sets | same as stage 1 (the benchmark's data model) |
