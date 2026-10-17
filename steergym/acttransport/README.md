# steergym.acttransport

Paper-faithful port of the Activation-Transport family:

- **Apple ml-act** (arXiv 2410.23054v2, "Activation Transport Quantifies
  Language Model Rewriting", code: github.com/apple/ml-act @ `d91642c`,
  Apple sample-code license, non-OSI — used/modified under its own terms,
  notice kept; cleanroom torch port, not vendored) — `MeanAcTModel` and
  `LinearAcTModel`.
- **PID-AcT** (Nguyen et al., the Mean-AcT fork at
  github.com/dungnvnus/pid-steering @ `d705c44`, MIT) — `PIDAcTModel`.

All three are `SteerModel`s whose fit is a SEQUENTIAL (causal) loop over an
arbitrary set of modules (not a single decoder layer) and whose context is
`MultiModuleTransformContext` (`context.py`). `steergym.odesteer.LinActModel`
is **not** touched by this package — see the delta table below.

## Provenance map

| piece | upstream file:line | our file |
|---|---|---|
| `MeanShiftMap` (pure mean shift) | `act/hooks/transport.py` `OnlyMeanHook` :273-307, `forward` :227-270 | `maps.py` |
| `IndependentSortOTMap` (linear OT) | `act/hooks/transport.py` `LinearOTHook` :506-528; `act/optimal_transport/ot_maps.py` `solve_ot_1d` :9-21; `act/optimal_transport/archs.py` `LinearProj.optimize` :30-54 | `maps.py` |
| support quantiles | `act/utils/quantiles.py` `compute_quantiles` | `maps.py` `support_bounds` |
| `find_module_names` | `act/models/model_with_hooks.py` `ModelWithHooks.find_module_names` :143-166 | `context.py` |
| sequential fit loop | `act/scripts/learn_intervention.py` `learn_intervention_all_incremental` :337-392, `TargetModuleReached` early abort | `models.py` `_sequential_fit` / `ModuleReached` |
| atonce ablation | `learn_intervention.py` `incremental == "atonce"` branch :420-437 | `models.py` `_atonce_fit` |
| PID law (released/degenerate) | `pidup/Mean-AcT/act/hooks/transport.py` `GaussianOTPIDHook`/`OnlyMeanPIDHook`: `fit` :404-412, `load_state_dict` scalar-integral :354-383, `forward` :484 | `models.py` `_pid_repo_compat_u` |
| PID law (canonical) | paper Def. 1 / Eq. 18, estimation per Eqs. 4-5 / 19 | `models.py` `_pid_fit_map` |

## Delta table: AcT paper protocol vs odesteer's `LinActModel` vs this submodule

| axis | AcT paper / ml-act code | `steergym.odesteer.LinActModel` | `steergym.acttransport.LinearAcTModel` |
|---|---|---|---|
| target modules | an arbitrary SET (Gemma-2 Post-LN: 52 norm modules; Llama `icml24`: mlp projections) | ONE decoder layer (`config.layer`) | a SET, resolved via `find_module_names` (registration order) |
| fit stats | per-sentence ATTENTION-MASKED MEAN over positions, SEQUENTIALLY captured with earlier modules' maps already installed | last-token activation (odesteer's extraction site) | per-sentence attention-masked mean, sequential (or `atonce`) |
| OLS fit | `w = Σ(x̃ȳ)/Σx̃²`, intercept `b = m_y − w·m_x`, SEEDED `1e-8` noise on `x`, means taken PRE-noise | `w = Σ(t·s)/Σ(s²+1e-10)`, mean-CENTERED sort, no noise, no intercept term kept separately (folded into `b`) | ml-act's exact formula (seeded noise, pre-noise means, no `+eps`) |
| pairing | equal-n per-unit independent sort (balanced upstream in the dataloader, ALL hook types) | truncate-pair per-unit sort | equal-n: `AcTTransportModel.fit` shuffles+truncates pos/neg TEXTS to `min(n)` (stdlib `random.Random(seed)`, not upstream's numpy `rng.permutation` — disclosed RNG delta, see GAP 6 below) before any pass; `IndependentSortOTMap.fit`'s own prefix truncation is now a residual guard |
| support masking | strict quantile mask (`q_lo < z < q_hi`), configurable key, computed from the SOURCE group | none | strict quantile mask, `support` config knob (`q_all`\|`q_0_100`\|...) |
| std masking | **Mean only**: units with `std ≤ 1e-4` on either group never transported; Linear-AcT (`LearnableOTHook`) has NO std mask, quantile support only | none | same split: `MeanAcTModel` gated by `std_eps`; `LinearAcTModel`'s mask is wide-open and `std_eps` is inert (see `LinearAcTConfig`) |
| application | `z' = strength·T(z) + (1-strength)·z` inside the mask, positions "all"\|"last" | `f(h)` applied unconditionally at every position (or `T`-strength for ODE-family methods only; LinAcT ignores `T`) | `z' = lmbda·T_k(z) + (1-lmbda)·z` inside the mask, positions "all"\|"last"\|"generated" |
| dose vocabulary | `strength` | `T` (ignored by LinAcT) | `lmbda` (Mean/Linear); `alpha` (PID) |

The two `Linear*` maps are deliberately different objects for a reason:
odesteer's `LinearTransportMap` is a single-decoder-layer, last-token-view,
truncate-pair ablation baseline inside a DIFFERENT paper's benchmark
(ODESteer, ICLR 2026); `IndependentSortOTMap` here is the literal AcT paper
protocol. Comparing the two IS a clean ablation (extraction site + support
masking + fit-noise regularization), the same way odesteer's CAA vs
DiffMean comparison is one — but they are not meant to be interchangeable.

## The paper-typo note

AcT paper Def 3.1 prints the OLS slope's denominator as `Σ(ỹ)²`. The
released code (`archs.py:41-43`) computes `Σ(x̃)²` — the textbook OLS
closed form (`beta = Cov(x,y)/Var(x)`), and is what we port. The typo
formula gives a measurably different slope whenever `std(target) !=
std(source)` (confirmed numerically in `test_acttransport.py`, section 1d:
on a fixture with `std(target) ≈ 1.5·std(source)`, the two formulas'
weights disagree by ~3x — a test built with the typo formula's denominator
in place of the code's would fail).

## The PID released-code degeneracy note

The fork's own toxicity-run code (`pidup/Mean-AcT`) does NOT implement the
paper's Def. 1 / Eq. 18 PID law. Tracing `act/hooks/transport.py`:

- `GaussianOTPIDHook.fit` (:404-412) sets `diff = mu2 - mu1 = r(k)` — a
  plain per-module mean difference, no PID structure yet.
- `GaussianOTPIDHook.load_state_dict` (:354-383, called right before the
  hook is used for a LATER module's incremental capture) globs every
  `.statedict` file in the output directory, stopping at the current
  module's own file — i.e. "earlier modules only", but in **`Path.glob`'s
  filesystem directory order**, which is UNSORTED and platform/filesystem-
  dependent (not lexicographic, and not fit-time incremental order either).
  So the subset this picks up can silently disagree with the actual fit
  order on any given run — a stale/reordering hazard the released code
  never guards against.
  It then concatenates those earlier modules' saved `diff` ([d] 1-D
  tensors) along `dim=0` — which, for 1-D tensors, FLATTENS them into one
  long vector mixing every coordinate of every earlier layer together —
  and takes `torch.mean(..., dim=0)` of THAT, collapsing the "integral"
  term to a single GLOBAL SCALAR `s_k`, with zero per-coordinate or
  per-layer structure:

      diff_m = diff + 0.005 * (s_k + diff) = 1.005 * diff + 0.005 * s_k

- `OnlyMeanPIDHook.forward` (:484, `hook_onlymean=True`) applies
  `z_ot = z + 0.7 * diff_m`, i.e.

      u(k) = 0.7 * diff_m = 0.7035 * r(k) + 0.0035 * s_k * ones(d)

  — no derivative term at all (`Kd` is absent from the released law).

`PIDAcTConfig(repo_compat=True)` reproduces exactly this formula
(`models.py`'s `_pid_repo_compat_u`), using REGISTRATION order (`j < k`,
exclusive) as a deterministic stand-in for the glob-order subset — the
approximation this flag discloses, not the literal filesystem-ordering
bug. `repo_compat=False` (default) is the canonical paper law:
`u(k) = Kp·r(k) + Ki·Σ_{j<k} r(j) + Kd·(r(k) − r(k-1))`, fully
per-coordinate, with a real `Kd` term (`r(-1) := 0` at `k=0` gives
`u(0) = (Kp + Kd)·r(0)`). The alternative convention `r(-1) := r(0)` (so
the D-term is zero at `k=0` — what the fork's own notebook variant does)
is noted here, not taken; `r(-1) := 0` is this port's choice.

Std masking: the canonical law is **unconditional** — `PIDShiftMap`'s
`mask` is wide-open, matching the paper's Def. 1 (no std gate at all). The
RELEASED fork code, however, only ever applies its degenerate law on
std-masked units (`OnlyMeanPIDHook`/`GaussianOTPIDHook` share
`GaussianOTHook`'s `std1 > eps & std2 > eps` gate) — `repo_compat=True`
reproduces that too: the fitted `PIDShiftMap` carries the same
`std_mask(std1, std2, std_eps)` gate as `MeanShiftMap`, computed from that
module's own pos/neg pools.

Also disclosed (not reproduced — it's trivia about the fork's OWN
Mean-AcT *baseline* comparison point, a different hook entirely): the same
file's non-PID `OnlyMeanHook.forward` is patched to
`z_ot = (z - 1.2·mu1) + 1.2·mu2` — a silent 1.2× rescale of both means —
and the fork's own scripts run Linear-AcT at `strength=5.0` inside the
sequential passes with `seed=38`. This would explain the published
Mean-AcT-sequential numbers (0.68) exceeding the AcT paper's own (0.54):
whether the published PID tables came from exactly this code snapshot is
GAP 4 below, and is left soft per the spec (no certification claim rides
on `repo_compat` matching a specific published number).

## Recorded GAPs (not blocking)

1. Jigsaw `n` actually used by the AcT paper ("hundreds") vs the code's
   all-balanced default — unresolved, affects nothing here (fit set size
   is a runscript/dataset concern).
2. AcT's TQA fit set: no code, paper silent.
3. The paper's Llama "Attention" module set (probably `self_attn.o_proj`)
   is unshipped in `MODULE_PRESETS` — not guessed at; `icml24` (mlp
   projections) is the only Llama preset shipped.
4. Whether the PID paper's published toxicity tables came from this exact
   code snapshot (the 1.2× Mean-AcT rescale + strength-5.0 Linear-AcT
   noted above) is unconfirmed — `repo_compat` numbers should be read as
   "this code's law", not "the published numbers", until checked.
5. `MeanAcTConfig`'s shipped default (`mean_ot.yaml`'s `support="q_all"`,
   i.e. no masking) diverges from the paper's own stated mitigation rule
   (`Q_o`, a real quantile support) — both values are exposed via the
   `support` knob; the yaml default is kept as OUR default too, since it's
   what the released artifacts/behavior actually used.
6. ~~`IndependentSortOTMap`'s "equal n" balancing is unshipped~~ CORRECTED:
   upstream's balanced subsample IS shipped (`learn_intervention.py`
   :285-291 → `datasets/responses_io.py` `label_src_dst_subsets`,
   `balanced=True`: a seeded permutation then truncation of BOTH groups to
   `min(n_pos, n_neg)`, for EVERY hook type including Mean, not just
   Linear). `AcTTransportModel.fit` now ports this semantically — right
   after `_contrast_texts`, both text lists are shuffled with
   `random.Random(config.seed)` and truncated to the shared minimum, before
   any capture pass runs (killing the wasted oversized-pool passes too).
   Our RNG (stdlib `random.Random.shuffle`) differs from upstream's numpy
   `rng.permutation`, so the realized subset is NOT bit-matched — a
   disclosed delta, not a bitwise port. `IndependentSortOTMap.fit`'s own
   `source_X[:n], target_X[:n]` prefix truncation remains as a residual
   guard (both inputs are already balanced by the time it runs; it only
   bites if a caller feeds it unbalanced pools directly).
7. No published PID Table-1 `Kp`/`Ki`/`Kd` values exist; `PIDAcTConfig`
   defaults to the paper's Fig. 6 sweet spot (`kp=1.0, ki=0.056, kd=0.01`).

## Cost note: O(L²) layer evaluations, per concept

The sequential fit makes **one capture pass per module per branch**
(`2·L` passes total for `L` target modules), each cut short right after
that module's own output via the `ModuleReached` early-abort hook — but
each pass still runs every layer UP TO its own target module k, i.e. `k+1`
layer evaluations, not O(1). Summed over `k = 0..L-1`, one branch costs
`sum_k (k+1) = L(L+1)/2 = O(L²)` layer evaluations, i.e. **~L/2
full-forwards-equivalent per branch** across the whole module chain; both
branches together are ~L full-forwards-equivalent. That is substantially
more than a single-layer method's one-pass fit (DiffMean, CAA, ...) or even
the `atonce` ablation's `2` total (always-full) passes
(`config.sequential=False`) — atonce trades this cost for giving up the
causal coupling between modules (every module's map is fit on RAW,
unperturbed activations, as if every earlier map were the identity). This
cost is **PER CONCEPT**: `fit()` re-runs the whole sequential (or atonce)
loop, including the negative pool, from scratch for every concept — there
is no cross-concept caching of the negative-branch captures.

## Defaults vs upstream (tier-1 parity expectation: close, not bit-exact)

`AcTTransportConfig`'s shipped defaults diverge from upstream's own run
configuration in ways that affect fit-stats precision, not correctness:
`max_length=1024` vs upstream's Gemma sequence-length truncation at `128`;
`batch_size=6` vs upstream's `32`; and our pooled-mean extraction is a
float32, masked mean over the captured activations, vs upstream's
`torch.nanmean` computed IN THE MODEL'S OWN DTYPE (bf16 for the paper's
runs). None of these are bugs — they are reasonable library defaults — but
together they mean a from-scratch steergym fit should be expected to land
CLOSE to a reproduction of the paper's numbers, not bit-identical to it;
only the `ACT_DIR`-gated upstream-source-parity leg (section 6 of
`test_acttransport.py`) checks bit/formula-level agreement, and it does so
on tensors shared between both implementations, not through a full
end-to-end fit at the paper's own hyperparameters.

## Persistence

One torch checkpoint per model: `config` + `module_names` (the realized,
registration-order module list from fit time) + per-concept, per-module
fitted-map `state_dict()`s. `fitted_buffers` pins the exact buffer key set
each class's map persists (`MeanAcTModel`: `mu1, mu2, mask, q_lo, q_hi`;
`LinearAcTModel`: `w, b, mask, q_lo, q_hi`; `PIDAcTModel`: `u, r, mask,
q_lo, q_hi` — `r` carried for provenance, not used at steer time); `load()`
rejects an artifact whose per-module buffer keys don't match. `context()`
re-resolves `find_module_names` against the bound model and rejects a
mismatch against the artifact's recorded `module_names` — a wrong model
(different layer count, different HF architecture under the same regex)
fails loudly instead of silently steering the wrong modules.
