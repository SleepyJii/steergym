# steergym.ifeval

IFEval verifiable instructions as steering concepts: the official checkers
(Zhou et al. 2023) as evaluators, Stolfo et al.'s (2024) instruction vectors
as a precomputed-row model, and the type registry / pack building / data
generation as `IFEvalBenchmark`.

## Benchmark

`IFEvalBenchmark(IFEvalConfig)` — one concept per Stolfo-pool `type_id`;
`Concept.name` = `Concept.description` = `build_description` at the pack's
modal effective kwargs (name is what the axbench fits filter on, description
is what `PromptSteerModel(placement="append")` renders — `base\ndesc` at
eval time; train-time constrained prompts are space-joined). `types.py`
builds per-type packs param-driven (`type_set`/`n_types`/`n_train`/
`n_eval`/`seed`); the Stolfo pool (8MB jsonl, Apache-2.0) is vendored under
`vendor/stolfo/` and overridable via `config.stolfo_dir`.

Stages: `concepts()`/`eval_samples()`/`stolfo_samples()` build locally from
the pool (no GPU, no dump). `train_samples()`/`polar_train_samples()`/
`latent_samples()` need a gen_pairs-layout dump — cached via `dump_dir`
(`packs.json` in the dump is the provenance record and wins over config type
params), or generated there by `bind_generator(GeneratorLM)` (the checker
filters constrained responses to genuine positives; plain per-unique-base
responses are the losing/negative side and the global `-1` pool;
`pass_rates.json` = the prompt-following baseline). StolfoDiffMean training
is `StolfoDiffMeanModel.fit(concepts, benchmark.stolfo_samples())`.

## Evals

| class | columns | notes |
|---|---|---|
| `IFEvalEval` | `pass_strict`, `pass_loose`, `checker_error` | official checkers (vendored, Apache-2.0); strict + IFEval's loose variants |
| `StolfoPerplexityEval(PerplexityEvalConfig(model_id, threshold))` | `ppl`, `low_ppl` | Stolfo's gate, verbatim mechanics; model_id = any transformers CausalLM (default gpt2, their 2.5 format / 4 word-specific threshold) |
| `StolfoQualityEval(QualityEvalConfig(judge_model))` | `quality`, `quality_frac_na` | their two-stage judge (App. F); API spend, never on by default |

Checker addressing rides in `meta.checker` on eval rows — the payload shape
(`CHECKER_META_KEY`, `{"instruction_id", "kwargs_json"}`) is defined HERE,
not in the shared schema: this submodule is its only producer/consumer
(meta keys are benchmark-local except `label`). kwargs must be
the *effective* pack-build-time kwargs: `build_description` randomises unset
ones.

## Models

`StolfoDiffMeanModel(AxbenchRowModel)` — rows precomputed by the data job's
own GPU pass; weight source is config (`dump_dir`), like every AxbenchRowModel.
Open problem, no answer yet: flat dose response at fixed L20 under max-act
scaling — their per-type layer search + projection-matched
weight rule are unimplemented, and precomputed rows don't recover them.

## Adaptation deltas vs the source implementations

- Per-row error flag (`checker_error`) instead of a module-global error
  count; checker/judge errors are `None` in the column, not silent zeros.
- Per-row columns only; macro/pooled aggregation is a runscript groupby.
- Thresholds and judge model ids are EvalConfig fields, not env knobs.
