# steergym.personavec

persona_vectors (Chen et al. 2025,
[arXiv:2507.21509](https://arxiv.org/abs/2507.21509), 7 published traits:
apathetic, evil, hallucinating, humorous, impolite, optimistic, sycophantic)
in the steergym `Eval`/`SteerModel`/`Benchmark` contracts. `TraitEval` and
`CoherenceEval` reproduce upstream's own published-number judge mechanics
exactly (real API spend). `PersonaVectorModel` is a zero-train
`DiffMean`-shaped row model (a closed-form activation-difference pass, no
gradient training).

## Upstream protocol pin

- Vendored at `vendor/persona_vectors/` (Apache-2.0), commit
  `UPSTREAM_COMMIT`. Only `reference/eval/prompts.py` is copied here
  (dependency-free, imported directly via `importlib.util.spec_from_file_location`
  so `Prompts["coherence_0_100"]` is byte-identical to upstream) plus
  `LICENSE`. Trait data JSONs (`trait_data_extract/`, `trait_data_eval/`) are
  **not** vendored here -- they belong to a later benchmark-adapter step
  (the `Concept` registry + `gen_traits`-equivalent data job).
- Judge model: `gpt-4.1-mini-2025-04-14` (`reference/eval/eval_persona.py:137`).
- Judge call: sync `openai.OpenAI` (not upstream's `AsyncOpenAI` -- steergym
  scores rows serially), `max_tokens=1, temperature=0, logprobs=True,
  top_logprobs=20, seed=0` (`reference/judge.py:49-70`).
- Score: probability-weighted mean over the top-20 first-token logprobs,
  restricted to integer keys 0-100; `None` (refusal) when the summed weight
  on valid number tokens is below 0.25 (`reference/judge.py:85-103`,
  `aggregate_0_100_score` in `evals.py`).
- Trait judge prompt = the trait's definition paragraph + a verbatim shared
  suffix (`[QUESTION START]...[ANSWER START]...Respond with a number between
  0 and 100...`). The suffix is identical across all 7 upstream traits'
  `trait_data_eval/*.json` `eval_prompt` fields (verified byte-identical at
  pack-build time) and is reproduced verbatim as
  `TRAIT_PROMPT_SUFFIX` in `evals.py`, since the JSONs themselves aren't
  vendored here yet.
- Coherence judge prompt = `Prompts["coherence_0_100"]`, unmodified.

## What changed in the steergym adaptation

- **Eval contract**: `TraitEval`/`CoherenceEval` subclass `steergym.evals.base.Eval`
  (`execution = "api"`, `score(rows, concepts) -> {"pv_trait"|"pv_coherence": [...]}`)
  instead of axbench's `Evaluator.compute_metrics(data)`. Rows are a polars
  `pl.DataFrame` with `concept_id, prompt, response` (steergym's samples
  schema) instead of a pandas frame keyed by
  `{model_name}_steered_generation`/`factor`; per-row aggregation
  (mean/by-factor/refusal-rate) is the benchmark's `aggregate()` job now, not
  the evaluator's.
- **Trait text now arrives via `Concept.description`** (the definition
  paragraph, `pv_traits.trait_description`'s marker-split text) instead of a
  `PACKS_BY_CID[concept_id]["eval_prompt"]` dict populated by a driver global.
  Nothing in this package hardcodes trait text.
- **Logprob-aggregation arithmetic is a plain module-level function**
  (`aggregate_0_100_score`), unit-testable with zero deps (`test_personavec.py`
  calls it directly against hand-built logprob dicts, no `openai` import
  needed).
- **`PersonaVectorModel` is an `AxbenchRowModel`**: same `[C, d]` row shape as
  every axbench-cache method. `weight_name` is exactly the cache filename
  stem -- which `{stem}_weight.pt` file the axbench dump format stores/loads
  for this model, nothing more. Construction goes through
  `from_axbench_dump`/`from_config_source` (weight source = config:
  `dump_dir` xor `hf_repo`); unit-norm with raw norms preserved when
  `config.unit_norm=True` (--
  axbench's max-act dosing scales with `||v||`, so raw upstream rows
  overdosed steering by ~65-100x). `config.unit_norm=False` switches the
  loaded file to `PersonaVector_weight_raw.pt` (upstream raw-coefficient
  convention) -- pair with `context(strength=...)`, never
  `context(factor=...)`.
- **OpenAI client is lazy and per-instance**, not a module-level singleton --
  constructed on first `score()` call, honors `OPENAI_BASE_URL` (env or
  `PersonaJudgeEvalConfig.base_url`) for pointing at a local OpenAI-compatible server to
  avoid spend, and refuses loudly (`RuntimeError`) if `OPENAI_API_KEY` is
  unset.

## Benchmark (the adapter step, landed after the eval/model port)

`PersonaVectorsBenchmark(PersonaVectorsConfig)` — one concept per trait,
`Concept.name` = the trait word (what the axbench fits filter on),
`Concept.description` = the rubric's definition paragraph. `traits.py`
builds the trait packs param-driven; the trait JSONs are vendored
(`vendor/persona_vectors/trait_data_{extract,eval}/`, 88K, commit-pinned
alongside the prompts) and overridable via `config.trait_data_dir`.

Fully generative like ifeval: `concepts()`/`eval_samples()` build locally;
`train/polar/latent` need a gen_traits-layout dump (`dump_dir` cached, or
generated there via `bind_generator`). Extraction follows upstream's
rollout protocol: `n_per_question=10` sampled responses per (question ×
instruction) prompt at `extraction_temperature=1.0` (their coupling —
T=0.0 iff n=1 — is documented on the config, not silently applied;
single-rollout T=1.0 packs remain reproducible as an explicit override). The effective
filter pairs pos/neg by (question, instruction, sample), their aligned-CSV
mask; the plain arm stays single-rollout (chassis data for the contrastive
fits, not upstream protocol). Eval side: `eval_samples()` keeps one row per held-out question;
upstream scores 10 samples per question — that repetition belongs to the
scoring runscript. The effective-pair filter is
`config.filter`: `"openai"` (default) runs the faithful
`TraitEval`/`CoherenceEval` at upstream's 50/100 thresholds, `"none"` keeps
every pair.
The dump gains `extraction/extraction_data.parquet` (pos/neg/plain rows with
per-row scores + the effective mask) — the input to
`PersonaVectorModel.fit`, which replaces the data job's vector pass
(response-avg resid diff over effective pairs; unit/raw storage per
`config.unit_norm`). Dumps from earlier pipelines stay readable for everything
except `extraction_samples()`.

## Standing caveat

- Small local judges confound coherency with content they disapprove of
  (e.g. an "evil"-instructed arm reads as low-coherence in a way correlated
  with trait expression, not just fluency). The faithful API judges
  (`TraitEval`/`CoherenceEval`) are the arbiter for persona claims; any
  local-judge numbers (steergym.evals.judges) are internal-comparison only.

## Files

- `evals.py` -- `TraitEval`, `CoherenceEval`, `aggregate_0_100_score`,
  vendored-prompt loading. Module level = stdlib + polars only; `openai`
  imported lazily inside `_ensure_client`.
- `models.py` -- `PersonaVectorModel(AxbenchRowModel)`.
- `vendor/persona_vectors/` -- `reference/eval/prompts.py`, `LICENSE`,
  `UPSTREAM_COMMIT` only.
- `../tests/test_personavec.py` -- pure-python, run via
  `python3 steergym/tests/test_personavec.py` from the repo root (see header
  comment for the PYTHONPATH it needs); `openai` is never imported (the
  client is monkeypatched directly onto the `Eval` instance).
