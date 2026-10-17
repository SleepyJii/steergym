from dataclasses import dataclass

from ..activations import last_token_activations
from ..vector import RowVectorConfig, RowVectorModel


@dataclass
class LastTokenContrastConfig(RowVectorConfig):
    # extraction knobs, not paper hyperparameters (the fits are closed-form —
    # the paper's only training choice is sklearn defaults, see inverse_reg_c):
    # (odesteer calls the extraction site "EOS" — it is the text's last token)
    batch_size: int = 6     # forward batching only; fit output is batch-size
                            # invariant (smoke-checked). 6 mirrors DiffMeanConfig
    max_length: int = 1024  # truncation, mirrors DiffMean's
    seed: int = 42          # inert (nothing sampled); kept for harness uniformity


@dataclass
class CAAConfig(LastTokenContrastConfig):
    pass


@dataclass
class RepEConfig(LastTokenContrastConfig):
    pass


@dataclass
class ITIProbeConfig(LastTokenContrastConfig):
    inverse_reg_c: float = 1.0  # sklearn LogisticRegression's default C — the
                                # value odesteer inherits implicitly (it never
                                # sets C); the one real method hyperparameter


# ---------------------------------------------------------------- row reductions
def caa_row(pos_X, neg_X):
    """CAA (odesteer _caa.py): mean(pos) - mean(neg). Last-token vs
    response-pooled extraction is the only divergence from DiffMean."""
    return pos_X.float().mean(dim=0) - neg_X.float().mean(dim=0)


def repe_row(pos_X, neg_X):
    """RepE (odesteer _repe.py): PCA-1 of the positionally-paired diffs.
    Delta: sign oriented positive-ward (odesteer keeps sklearn svd_flip's)."""
    import torch

    n = min(len(pos_X), len(neg_X))
    diff = (pos_X[:n] - neg_X[:n]).float()
    centered = diff - diff.mean(dim=0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    row = vh[0]  # top right-singular vector == first PCA component
    if row @ diff.mean(dim=0) < 0:
        row = -row
    return row


def logistic_fit(pos_X, neg_X, inverse_reg_c=1.0):
    """(w [d], b scalar) of an L2-regularized logistic regression (sklearn's
    default objective: softplus sum + |w|^2/2C, intercept unpenalized),
    solved with full-batch LBFGS in float64 — convex, so the solver swap is
    exact to tolerance."""
    import torch

    X = torch.cat([pos_X, neg_X], dim=0).double()
    y = torch.cat([torch.ones(len(pos_X)), -torch.ones(len(neg_X))]).double()
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    # solver settings, not method hyperparameters: anything that converges the
    # convex objective yields the same optimum (stationarity is suite-checked)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=500,
                            tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        margins = y * (X @ w + b)
        loss = torch.nn.functional.softplus(-margins).sum() \
            + (w @ w) / (2.0 * inverse_reg_c)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().float(), float(b.detach())


def logreg_direction(pos_X, neg_X, inverse_reg_c=1.0):
    """ITI probe (odesteer _iti.py): the logistic probe's weight vector.
    Unnormalized; callers unit-norm."""
    w, _ = logistic_fit(pos_X, neg_X, inverse_reg_c)
    return w


# ---------------------------------------------------------------- shared fit input
def concept_contrast_activations(generator, tokenizer, samples, concept, config):
    """(pos_X [Np, d], neg_X [Nn, d]) — the per-concept last-token
    activation view every method in this package fits on (vector pack and
    transform family alike)."""
    from ..axbench.protocol import binarize_text, positive_and_negative_rows, prefix_suffix_lengths

    _, suffix_length = prefix_suffix_lengths(tokenizer)
    positive, negative = positive_and_negative_rows(samples, concept)
    pos_texts = [binarize_text(tokenizer, r["prompt"], r["response"], suffix_length)
                 for r in positive.iter_rows(named=True)]
    neg_texts = [binarize_text(tokenizer, r["prompt"], r["response"], suffix_length)
                 for r in negative.iter_rows(named=True)]
    pos_X = last_token_activations(generator, pos_texts, config)
    neg_X = last_token_activations(generator, neg_texts, config)
    return pos_X, neg_X


# ---------------------------------------------------------------- models
class _LastTokenContrastModel(RowVectorModel):
    """Abstract: fit = one _contrast_row per concept from last-token pos/neg
    activations; steering and persistence are the inherited RowVectorModel
    path (all-position addition, factor × max_act dose, torch checkpoint).
    weight_name = the artifact/log stem (no axbench dump interop — none of
    these methods has a released axbench dictionary)."""

    weight_name: str | None = None

    def _contrast_row(self, pos_X, neg_X):
        # the method's objective: one (unnormalized) steering row from
        # [Np, d] positive and [Nn, d] negative activations
        raise NotImplementedError

    def fit(self, concepts, samples, log=lambda msg: None):
        import time

        import torch
        from transformers import set_seed

        from ..generator import GeneratorLM

        model, tokenizer = self._require_bound()
        generator = GeneratorLM(self.config.subject, model, tokenizer)
        set_seed(self.config.seed)

        rows, raw_norms = [], []
        for concept in concepts:
            t0 = time.time()
            pos_X, neg_X = concept_contrast_activations(
                generator, tokenizer, samples, concept, self.config)

            row = self._contrast_row(pos_X, neg_X)
            # pre-norm scale recorded. Odesteer-native relevance is CAA ONLY:
            # their _caa.py keeps steer_vec unnormalized (published T=1 = the
            # raw mean-diff = unit row x raw_norm), while _iti.py and _repe.py
            # unit-normalize theirs — for those, odesteer-native dosing is the
            # unit row at swept strength T and raw_norms is just provenance
            raw_norms.append(float(row.norm()))
            row = row / (row.norm() + torch.finfo(row.dtype).eps)
            rows.append(row)
            log(f"{type(self).weight_name} concept {concept.id} ({concept.name!r}): "
                f"{len(pos_X)}+{len(neg_X)} rows, {time.time() - t0:.1f}s")

        self.rows = torch.stack(rows)
        self.bias = torch.zeros(len(rows), dtype=torch.float32)
        self.concept_index = {c.id: i for i, c in enumerate(concepts)}
        self.raw_norms = raw_norms
        return self


class CAAModel(_LastTokenContrastModel):
    weight_name = "CAA"
    config_cls = CAAConfig

    def _contrast_row(self, pos_X, neg_X):
        return caa_row(pos_X, neg_X)


class RepEModel(_LastTokenContrastModel):
    weight_name = "RepE"
    config_cls = RepEConfig

    def _contrast_row(self, pos_X, neg_X):
        return repe_row(pos_X, neg_X)


class ITIProbeModel(_LastTokenContrastModel):
    """odesteer's 'ITI' — a single-layer resid probe, not the paper's
    per-head intervention; named honestly to keep 'ITI' free."""

    weight_name = "ITIProbe"
    config_cls = ITIProbeConfig

    def _contrast_row(self, pos_X, neg_X):
        return logreg_direction(pos_X, neg_X, self.config.inverse_reg_c)
