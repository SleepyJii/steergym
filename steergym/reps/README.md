# steergym.reps

The RePS lineage from AxBench (Wu, Yu, Arora, Manning, Potts 2025,
arXiv:2505.20809) — RePS is the preference-pair training extension of the
ReFT-r1 line, both living in the axbench repo and sharing its dump format.

## Models

| class | weight stem | provenance |
|---|---|---|
| `LsReFTModel` | `LsReFT` | ReFT-r1; released `pyvene/gemma-reft-r1-2b-it-res` (`config.hf_repo`), or native `fit()` |
| `PreferenceVectorModel` | `PreferenceVector` | RePS rank-1; no released artifacts — `config.dump_dir` at a training dump, or native `fit()` |

Both are `AxbenchRowModel` (`steergym/axbench.py`): where weights come from
is config (`dump_dir` xor `hf_repo`), not baked into the class. `DiffMeanModel`
is not RePS-specific and lives with `AxbenchRowModel` itself.

Both models train natively: `LsReFTModel.fit` (TopKReLU objective) and
`PreferenceVectorModel.fit` (scaled-SimPO over orig_add/orig_sub preference
pairs), each delegating to a same-file private helper.
