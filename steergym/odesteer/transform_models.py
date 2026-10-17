"""Per-concept SteerModel wrappers for the odesteer transform family
(the pure transforms live in transforms.py): configs with the paper's conf
values as defaults (faithfulness doctrine; `lin_clf_type` is not a knob —
the paper runs 'lr' and that is what the torch fit implements), fits over
the same last-token pos/neg activation view as the vector pack, and the
inherited TransformSteerModel persistence.
"""
from dataclasses import dataclass

import torch

from ..transform import TransformSteerConfig, TransformSteerModel
from .models import concept_contrast_activations
from .transforms import (
    GatedGaussianTransportMap,
    LinearTransportMap,
    ODEMap,
    StepODEMap,
)


# ---------------------------------------------------------------- configs
@dataclass
class LastTokenTransformConfig(TransformSteerConfig):
    # extraction knobs, identical to the vector pack's
    # (LastTokenContrastConfig) — the fit input is the same per-concept
    # last-token pos/neg activation view
    batch_size: int = 6
    max_length: int = 1024
    seed: int = 42


@dataclass
class LinActConfig(LastTokenTransformConfig):
    pass


@dataclass
class MiMiCConfig(LastTokenTransformConfig):
    inverse_reg_c: float = 1.0  # sklearn LogisticRegression default C (gate)
    ot_reg: float = 1e-2        # odesteer: ot.da.LinearTransport(reg=1e-2)


@dataclass
class StepODESteerConfig(LastTokenTransformConfig):
    # paper conf (confs/steer/StepODESteer.yaml + appendix C.2)
    degree: int = 2
    n_components: int = 8000
    gamma: float = 0.1
    coef0: float = 1.0
    inverse_reg_c: float = 1.0  # sklearn default C, inherited implicitly upstream


@dataclass
class ODESteerConfig(StepODESteerConfig):
    # paper conf (confs/steer/ODESteer.yaml + appendix C.3)
    solver: str = "euler"
    steps: int = 10



# ---------------------------------------------------------------- models
class _LastTokenTransformModel(TransformSteerModel):
    """Abstract: fit = one transform per concept from the same last-token
    pos/neg activation view the vector pack fits on; steering and
    persistence are the inherited TransformSteerModel path."""

    def fit(self, concepts, samples, log=lambda msg: None):
        import time

        from transformers import set_seed

        from ..generator import GeneratorLM

        model, tokenizer = self._require_bound()
        generator = GeneratorLM(self.config.subject, model, tokenizer)
        set_seed(self.config.seed)

        self.transforms = {}  # wholesale rebuild, like RowVectorModel.fit
        for concept in concepts:
            t0 = time.time()
            pos_X, neg_X = concept_contrast_activations(
                generator, tokenizer, samples, concept, self.config)
            self.transforms[concept.id] = self._fit_transform(pos_X, neg_X)
            log(f"{type(self).weight_name} concept {concept.id} ({concept.name!r}): "
                f"{len(pos_X)}+{len(neg_X)} rows, {time.time() - t0:.1f}s")
        return self


class LinActModel(_LastTokenTransformModel):
    weight_name = "LinAcT"
    config_cls = LinActConfig
    fitted_buffers = frozenset({"w", "b"})

    def _new_transform(self):
        return LinearTransportMap()

    def _fit_transform(self, pos_X, neg_X):
        # their LinAcT.fit: lin_ot.fit(X_target=pos_X, X_source=neg_X)
        return LinearTransportMap().fit(target_X=pos_X, source_X=neg_X)


class MiMiCModel(_LastTokenTransformModel):
    weight_name = "MiMiC"
    config_cls = MiMiCConfig
    fitted_buffers = frozenset({"gate_w", "gate_b", "A", "b"})

    def _new_transform(self):
        return GatedGaussianTransportMap()

    def _fit_transform(self, pos_X, neg_X):
        return GatedGaussianTransportMap().fit(pos_X, neg_X,
                                        inverse_reg_c=self.config.inverse_reg_c,
                                        ot_reg=self.config.ot_reg)


class StepODESteerModel(_LastTokenTransformModel):
    weight_name = "StepODESteer"
    config_cls = StepODESteerConfig
    fitted_buffers = frozenset({"dre_coeff", "coef", "intercept",
                                "kernel.nFeat_", "kernel.indexHash_", "kernel.bitHash_"})

    def _check_loaded(self, transform):
        # the artifact's sketch state must agree with the config the module
        # was rebuilt from — a mismatch computes plausible-looking garbage
        # (or dies deep in vjp) instead of failing here
        k = transform.kernel
        if k.indexHash_.shape[0] != self.config.degree:
            raise ValueError(f"artifact degree {k.indexHash_.shape[0]} != "
                             f"config degree {self.config.degree}")
        if transform.coef.numel() != self.config.n_components:
            raise ValueError(f"artifact n_components {transform.coef.numel()} != "
                             f"config n_components {self.config.n_components}")

    def _new_transform(self):
        c = self.config
        return StepODEMap(degree=c.degree, n_components=c.n_components,
                          gamma=c.gamma, coef0=c.coef0)

    def _fit_transform(self, pos_X, neg_X):
        # hash draw seeded from config.seed: artifacts reproduce from config
        # alone (the realized buffers are saved regardless)
        return self._new_transform().fit(
            pos_X, neg_X, inverse_reg_c=self.config.inverse_reg_c,
            generator=torch.Generator().manual_seed(self.config.seed))


class ODESteerModel(StepODESteerModel):
    weight_name = "ODESteer"
    config_cls = ODESteerConfig

    def _new_transform(self):
        c = self.config
        return ODEMap(degree=c.degree, n_components=c.n_components,
                      gamma=c.gamma, coef0=c.coef0, solver=c.solver, steps=c.steps)
