from .models import (
    FLAS_CHECKPOINTS,
    CachedFLASConfig,
    CachedFLASContext,
    CachedFLASSteerModel,
    FLASCheckpointError,
)

# .flow_model (module-level torch/nn, nn.Module subclasses) stays unimported here
# -- a bare `import steergym` must not need torch (steergym.models imports
# this package, via models.py, which is torch-lazy). PEP 562 module
# __getattr__ keeps `from steergym.flas import FlowFunction` etc. working.
_FLOW_MODEL_EXPORTS = {
    "ArchSpec", "ConceptEncoder", "FlowBlock", "FlowCrossAttention", "FlowFunction",
    "TimeEmbedder", "get_arch_spec", "get_text_config", "get_text_decoder",
}


def __getattr__(name):
    if name in _FLOW_MODEL_EXPORTS:
        from . import flow_model

        return getattr(flow_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
