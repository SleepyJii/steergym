# steergym

Clean, unified re-implementations of published LLM activation-steering
methods and their benchmarks, behind one small shared interface.

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
- Shared machinery: `generator.py` (`GeneratorLM` subject loading +
  generation under contexts), `helpers.py` (`steer_sweep`),
  `activations.py`, `training_helpers.py`.

## Methods

| package | methods | paper |
|---|---|---|
| `reps/` | `LsReFTModel` (ReFT-r1), `PreferenceVectorModel` (RePS) | AxBench (Wu et al. 2025), RePS (Wu et al. 2025) |
| `axbench/` | `DiffMeanModel`, `AxbenchPromptSteerModel`, released-dictionary loading | AxBench (Wu et al. 2025) |
| `odesteer/` | `CAAModel`, `RepEModel`, `ITIProbeModel`, `LinActModel`, `MiMiCModel`, `ODESteerModel` | ODESteer (ICLR 2026) + its bundled baselines |
| `acttransport/` | `MeanAcTModel`, `LinearAcTModel`, `PIDAcTModel` | AcT (Rodriguez et al. 2024), PID-AcT (Nguyen et al.) |
| `alqr/` | `ALQRModel`, `SPIDModel` (closed-loop controllers) | A-LQR (2025), S-PID |
| `gcad/` | `GCADModel` (gated cropped attention-delta) | GCAD (2025) |
| `flas/` | `CachedFLASSteerModel` (published checkpoints, inference-only) | FLAS (NeurIPS 2026) |
| `personavec/` | `PersonaVectorModel` | Persona Vectors (Chen et al. 2025) |
| `ifeval/` | `StolfoDiffMeanModel` | Stolfo et al. 2024 |
| `models/` | `PromptSteerModel` | prompt-steering baseline |

Benchmarks: `Concept500Benchmark` (AxBench), `IFEvalBenchmark` (official
checkers vendored), `PersonaVectorsBenchmark`. Each subpackage README
records the upstream repo/commit, a provenance map (upstream file:line →
ours), and every adaptation delta.

Full paper list with per-class attribution: [papers.md](papers.md).

Judges: each benchmark's `evals()` returns its paper's own evals, trivially
constructed (AxBench's gpt-4o-mini 0-2 judges, IFEval checkers, Stolfo
perplexity/quality gates, persona_vectors' verbatim API judges). Every eval
declares its resources: `execution` (`local`/`gpu`/`api`) and, for API
judges, `api_key_env` — constructing one without that env var set emits a
`MissingAPIKeyWarning` up front rather than failing mid-run. A generic local
digit-token judge panel lives in `steergym/evals/judges.py` for API-free
scoring, assembled explicitly in runscripts and never baked into a paper
benchmark.

## Conventions

- `import steergym` works in a bare venv: module level = stdlib + polars;
  torch/transformers/openai import lazily inside functions.
- Aggregators `steergym.models` / `steergym.evals` re-export every method
  and eval.
- Tests under `steergym/tests/` are pure-python (no pytest):
  `python steergym/tests/test_<name>.py`.
