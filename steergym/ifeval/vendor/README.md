# vendored third-party files

- `ifeval_scripts/` — the official IFEval checkers (`instructions.py`,
  `instructions_registry.py`, `instructions_util.py`), Apache-2.0,
  google-research/instruction_following_eval, copied verbatim. Runtime
  deps: `absl-py langdetect nltk immutabledict` + NLTK `punkt`
  (`NLTK_DATA=$TMPDIR` on clusters — no home drive).
- `stolfo/` — Stolfo et al.'s augmented IFEval pool (Apache-2.0), the
  source `IFEvalBenchmark` builds packs from (`config.stolfo_dir`
  overrides).
