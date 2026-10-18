from .benchmark import (
    CONCEPT500_FACTORS,
    Concept500Benchmark,
    Concept500Config,
    canonical_samples,
)
from .models import (
    AxbenchPromptSteerContext,
    AxbenchPromptSteerModel,
    AxbenchRowConfig,
    AxbenchRowModel,
    DiffMeanConfig,
    DiffMeanModel,
)
from .protocol import (
    binarize_text,
    chat_prompt_text,
    latent_max_acts,
    latent_texts,
    positive_and_negative_rows,
    prefix_suffix_lengths,
    template_ids,
)

# evals re-exports are lazy (PEP 562): the root package imports axbench FIRST,
# and axbench/evals.py pulls the steergym.evals aggregator (via evals.base),
# which imports ..axbench.evals back while this package is initializing — the
# documented aggregator circularity, evals flavor
_EVALS = ("AxbenchConceptEval", "AxbenchFluencyEval", "AxbenchInstructEval",
          "AxbenchJudgeConfig", "harmonic_mean_strict", "lm_judge_rating", "parse_rating")


def __getattr__(name):
    if name in _EVALS:
        from . import evals
        return getattr(evals, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EVALS))
