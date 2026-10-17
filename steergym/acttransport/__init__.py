from .context import MODULE_PRESETS, ModuleReached, MultiModuleTransformContext, find_module_names, resolve_module_regexes
from .models import (
    AcTTransportConfig,
    LinearAcTConfig,
    LinearAcTModel,
    MeanAcTConfig,
    MeanAcTModel,
    PIDAcTConfig,
    PIDAcTModel,
)

# .maps (module-level torch, nn.Module subclasses) stays unimported here --
# a bare `import steergym` must not need torch (steergym.models imports
# this package). PEP 562 module __getattr__ keeps `from steergym.acttransport
# import IndependentSortOTMap` etc. working lazily.
_MAP_EXPORTS = {"IndependentSortOTMap", "MeanShiftMap", "PIDShiftMap", "std_mask", "support_bounds"}


def __getattr__(name):
    if name in _MAP_EXPORTS:
        from . import maps

        return getattr(maps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
