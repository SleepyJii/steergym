# steergym

Clean, unified re-implementations of published LLM activation-steering
methods and their benchmarks, behind one small shared interface.

## Contents

### Methods

| paper | submodule | method | descr |
|---|---|---|---|
| [AxBench](https://arxiv.org/abs/2501.17148) (Wu et al. 2025) | `reps/` | `LsReFTModel` | ReFT-r1 rank-1 rows, TopKReLU training |
| AxBench | `axbench/` | `DiffMeanModel` | difference-of-means rows, closed form |
| AxBench | `axbench/` | `AxbenchPromptSteerModel` | the paper's prompt-steering arm |
| [RePS](https://arxiv.org/abs/2505.20809) (Wu et al. 2025) | `reps/` | `PreferenceVectorModel` | rank-1 rows, scaled-SimPO preference training |
| [ODESteer](https://github.com/ZhaoHongjue/odesteer) (ICLR 2026) | `odesteer/` | `ODESteerModel`, `StepODESteerModel` | ODE flow along a learned density-ratio field |
| CAA (Rimsky et al. 2024) | `odesteer/` | `CAAModel` | mean-difference contrast vector |
| RepE (Zou et al. 2023) | `odesteer/` | `RepEModel` | PCA-1 of paired contrast diffs |
| ITI probe (as bundled in ODESteer) | `odesteer/` | `ITIProbeModel` | logistic-probe direction |
| LinAcT / MiMiC (as bundled in ODESteer) | `odesteer/` | `LinActModel`, `MiMiCModel` | linear / gated Gaussian-OT transport transforms |
| [AcT](https://arxiv.org/abs/2410.23054) (Rodriguez et al. 2024) | `acttransport/` | `MeanAcTModel`, `LinearAcTModel` | sequential multi-module transport maps |
| [PID Steering](https://arxiv.org/abs/2510.04309) (Nguyen et al.) | `acttransport/` | `PIDAcTModel` | per-coordinate PID setpoint control |
| [A-LQR](https://arxiv.org/abs/2604.19018) | `alqr/` | `ALQRModel` | per-layer LQR feedback tracking a contrastive setpoint |
| A-LQR (its own ablation) | `alqr/` | `SPIDModel` | S-PID setpoint controller |
| [GCAD](https://arxiv.org/abs/2605.10664) | `gcad/` | `GCADModel` | gated cropped attention-delta steering |
| [FLAS](https://arxiv.org/abs/2605.05892) (NeurIPS 2026) | `flas/` | `CachedFLASSteerModel` | published flow checkpoints, inference-only |
| [Persona Vectors](https://arxiv.org/abs/2507.21509) (Chen et al. 2025) | `personavec/` | `PersonaVectorModel` | zero-train trait rows |
| [Stolfo et al. 2024](https://arxiv.org/abs/2410.12877) | `ifeval/` | `StolfoDiffMeanModel` | last-token diff-mean instruction vectors |
| — (common baseline) | `models/` | `PromptSteerModel` | prompt steering as a peer method |

### Benchmarks

| paper | benchmark | descr |
|---|---|---|
| AxBench | `Concept500Benchmark` | 500 SAE concepts, AlpacaEval steering prompts, latent max-act dosing pass |
| IFEval (Zhou et al. 2023) + Stolfo et al. 2024 | `IFEvalBenchmark` | verifiable instructions as concepts; generative pack building from the vendored augmented pool |
| Persona Vectors | `PersonaVectorsBenchmark` | the 7 published traits as concepts; upstream rollout + effective-pair filter protocol |

### Evals

| eval | benchmark | descr |
|---|---|---|
| `AxbenchConceptEval`, `AxbenchInstructEval`, `AxbenchFluencyEval` | Concept500 | the leaderboard's gpt-4o-mini 0-2 judges, verbatim templates; strict-harmonic-mean aggregate via `lm_judge_rating` |
| `IFEvalEval` | IFEval | official vendored checkers, strict + loose |
| `StolfoPerplexityEval` | IFEval | GPT-2 perplexity degeneracy gate |
| `StolfoQualityEval` | IFEval | the paper's two-stage LM quality judge (opt-in API spend) |
| `TraitEval`, `CoherenceEval` | PersonaVectors | the paper's own 0-100 logprob-aggregated API judges |
| `CoherencyEval`, `MentionsEval`, `ExpressionEval` (`evals/judges.py`) | — (generic) | local digit-token judge panel, API-free; assembled in runscripts, never returned by a benchmark |

Each method subpackage README records the upstream repo/commit, a provenance
map (upstream file:line → ours), and every adaptation delta. Full paper list
with per-class attribution: [papers.md](papers.md).

## The object model

Four peer families, all `Thing(config)` with one dataclass config per
subclass (`config_cls`); live resources (a loaded subject model, a GPU
judge, an API client) are bound via `bind_*` methods, never config — so a
saved artifact's config is its provenance.

- **`SteerModel`** (`steergym/steer.py`) — `fit(concepts, samples)` trains
  (or loads precomputed artifacts), `save`/`load` persist, and
  `context(concept, **coeffs)` returns a **`SteerContext`**: a context
  manager over the subject model exposing forward hooks and/or a prompt
  `render` seam, both no-op by default. Prompt steering is an ordinary
  peer method. Dose/coefficient vocabulary is method-owned (`factor`,
  `strength`, `T`, `lmbda`, ...) — cross-method comparability comes from
  `SteerContext.diagnostics()` (applied ‖Δh‖), not a universal dose schema.
- **`Benchmark`** (`steergym/benchmark.py`) — defines DATA: `concepts()`
  plus sample views over one canonical schema (`concept_id, prompt,
  response?, meta`). Benchmarks have no sweep/aggregate methods;
  coefficient grids and groupbys are runscript concerns.
- **`Eval`** (`steergym/evals/base.py`) — `score(rows, concepts)` returns
  per-row columns aligned to the results table; aggregation is a groupby.
  A benchmark's `evals()` returns its paper's own evals, trivially
  constructed. Every eval declares its resources: `execution`
  (`local`/`gpu`/`api`) and, for API judges, `api_key_env` — constructing
  one without that env var set emits a `MissingAPIKeyWarning` up front
  rather than failing mid-run.
- Shared machinery: `generator.py` (`GeneratorLM` subject loading +
  generation under contexts), `helpers.py` (`steer_sweep`),
  `activations.py`, `training_helpers.py`.

## Conventions

- `import steergym` works in a bare venv: module level = stdlib + polars;
  torch/transformers/openai import lazily inside functions.
- Aggregators `steergym.models` / `steergym.evals` re-export every method
  and eval.
- Tests under `steergym/tests/` are pure-python (no pytest):
  `python steergym/tests/test_<name>.py`.
