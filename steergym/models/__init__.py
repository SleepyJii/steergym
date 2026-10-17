from ..acttransport.models import LinearAcTModel, MeanAcTModel, PIDAcTModel
from ..alqr import ALQRModel, SPIDModel
from ..axbench import AxbenchPromptSteerModel, AxbenchRowConfig, AxbenchRowModel, DiffMeanModel
from ..flas.models import CachedFLASConfig, CachedFLASSteerModel, FLASCheckpointError
from ..gcad.models import GCADModel
from ..ifeval.models import StolfoDiffMeanModel
from ..personavec.models import PersonaVectorModel
from .prompt_steerer import PromptSteerConfig, PromptSteerContext, PromptSteerModel
from ..reps.models import LsReFTModel, PreferenceVectorModel
from ..steer import SteerModel
from ..transform import TransformSteerConfig, TransformSteerModel
from ..vector import RowVectorConfig, RowVectorModel

# odesteer re-exports are lazy (PEP 562): the odesteer package defines torch
# Modules (sketch/transforms) at module level, and `import steergym` must
# work in a bare venv (stdlib + polars only)
_ODESTEER_MODELS = ("CAAModel", "ITIProbeModel", "LinActModel", "MiMiCModel",
                    "ODESteerModel", "RepEModel", "StepODESteerModel")


def __getattr__(name):
    if name in _ODESTEER_MODELS:
        from .. import odesteer
        return getattr(odesteer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_ODESTEER_MODELS))
