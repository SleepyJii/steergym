#!/usr/bin/env python3
# Offline tests for steergym.odesteer's row reductions and config surface.
# Unlike test_training.py this suite NEEDS torch (SVD / LBFGS are the
# reductions under test) — CPU torch is enough. Run:
#   PYTHONPATH=<pylibs>:<torchlibs> python3 steergym/tests/test_odesteer.py
# The fake-LM end-to-end fits (extraction included) live in the devtools
# smoke harness (smoke_odesteer.py), not here.
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from steergym.generator import GeneratorRef
from steergym.vector import RowVectorConfig, RowVectorModel
from steergym.odesteer import (
    CAAConfig,
    CAAModel,
    ITIProbeConfig,
    ITIProbeModel,
    LastTokenContrastConfig,
    RepEConfig,
    RepEModel,
    caa_row,
    logreg_direction,
    repe_row,
)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


SUBJECT = GeneratorRef("fake/subject")


# ---------------------------------------------------------------- config defaults (provenance)
cfg = LastTokenContrastConfig(subject=SUBJECT)
check("LastTokenContrastConfig: batch_size=6", cfg.batch_size == 6)
check("LastTokenContrastConfig: max_length=1024", cfg.max_length == 1024)
check("LastTokenContrastConfig: seed=42", cfg.seed == 42)
check("LastTokenContrastConfig: is a RowVectorConfig", isinstance(cfg, RowVectorConfig))
check("ITIProbeConfig: inverse_reg_c=1.0 (sklearn C default)",
      ITIProbeConfig(subject=SUBJECT).inverse_reg_c == 1.0)

check("CAAModel: config_cls is CAAConfig", CAAModel.config_cls is CAAConfig)
check("RepEModel: config_cls is RepEConfig", RepEModel.config_cls is RepEConfig)
check("ITIProbeModel: config_cls is ITIProbeConfig",
      ITIProbeModel.config_cls is ITIProbeConfig)
check("weight stems: CAA / RepE / ITIProbe",
      (CAAModel.weight_name, RepEModel.weight_name, ITIProbeModel.weight_name)
      == ("CAA", "RepE", "ITIProbe"))
check("models are RowVectorModels with nothing axbench in the MRO",
      all(issubclass(m, RowVectorModel)
          and all("Axbench" not in c.__name__ for c in m.__mro__)
          for m in (CAAModel, RepEModel, ITIProbeModel)))


# ---------------------------------------------------------------- caa_row
pos = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
neg = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
got = caa_row(pos, neg)
check("caa_row: mean(pos) - mean(neg) by hand",
      torch.allclose(got, torch.tensor([0.0, 1.0, 2.0])))
check("caa_row: unequal set sizes are fine (means, not pairs)", got.shape == (3,))


# ---------------------------------------------------------------- repe_row
g = torch.Generator().manual_seed(0)
d = 16
u = torch.zeros(d)
u[3] = 1.0  # the dominant variance direction we plant
base = torch.randn(40, d, generator=g)
spread = torch.randn(40, 1, generator=g) * 25.0
pos_X = base + spread * u + 0.6 * u  # diffs vary hugely along u, small mean shift
neg_X = torch.randn(40, d, generator=g) * 0.1

row = repe_row(pos_X, neg_X)
check("repe_row: unit norm (SVD right-singular vector)",
      abs(row.norm().item() - 1.0) < 1e-5)
check("repe_row: recovers the planted dominant direction",
      abs(abs(row[3].item()) - 1.0) < 1e-2)
diff_mean = (pos_X - neg_X).mean(dim=0)
check("repe_row: oriented positive-ward (dot with mean diff >= 0)",
      (row @ diff_mean).item() >= 0)
row_swapped = repe_row(neg_X, pos_X)
check("repe_row: swapping pos/neg flips the row (same subspace, orientation tracks)",
      torch.allclose(row, -row_swapped, atol=1e-4))

# truncation to the shorter set, in row order
pos_long = torch.cat([pos_X, 1000.0 * torch.ones(5, d)])  # rows past n must be ignored
row_trunc = repe_row(pos_long[: len(neg_X) + 5], neg_X)
check("repe_row: truncates to min length in row order (extra pos rows ignored)",
      torch.allclose(row_trunc, row, atol=1e-4))


# ---------------------------------------------------------------- logreg_direction
g2 = torch.Generator().manual_seed(1)
sep = torch.zeros(6)
sep[1] = 2.0
pos_c = torch.randn(30, 6, generator=g2) + sep
neg_c = torch.randn(30, 6, generator=g2) - sep
w = logreg_direction(pos_c, neg_c)
check("logreg_direction: points from neg toward pos",
      (w @ (pos_c.mean(0) - neg_c.mean(0))).item() > 0)
check("logreg_direction: dominant coordinate is the separating one",
      w.abs().argmax().item() == 1)

# stationarity of the exact sklearn-default objective at the returned optimum:
# grad_w = -sum_i y_i * sigmoid(-y_i m_i) * x_i + w / C ~= 0
X = torch.cat([pos_c, neg_c]).double()
y = torch.cat([torch.ones(30), -torch.ones(30)]).double()
# recover the intercept the fit found by re-solving 1-d in b at fixed w (cheap
# bisection is overkill: re-run the fit and trust determinism, checked below)
w2 = logreg_direction(pos_c, neg_c)
check("logreg_direction: deterministic (two fits identical)", torch.equal(w, w2))
wd = w.double()
b_grid = torch.linspace(-3, 3, 6001, dtype=torch.float64)
losses = torch.stack([
    torch.nn.functional.softplus(-y * (X @ wd + b)).sum() for b in b_grid])
b_star = b_grid[losses.argmin()]
margins = y * (X @ wd + b_star)
grad_w = -(y * torch.sigmoid(-margins)) @ X + wd / 1.0
check("logreg_direction: stationary point of the L2-regularized objective "
      f"(|grad| {grad_w.norm().item():.2e})", grad_w.norm().item() < 1e-3)

w_loose = logreg_direction(pos_c, neg_c, inverse_reg_c=100.0)
check("logreg_direction: weaker regularization (C=100) -> larger weights",
      w_loose.norm().item() > w.norm().item())

# the models dispatch their reductions (no binding needed for _contrast_row)
m_caa = CAAModel(CAAConfig(subject=SUBJECT))
check("CAAModel._contrast_row == caa_row", torch.allclose(m_caa._contrast_row(pos, neg), caa_row(pos, neg)))
m_iti = ITIProbeModel(ITIProbeConfig(subject=SUBJECT, inverse_reg_c=100.0))
check("ITIProbeModel._contrast_row passes inverse_reg_c through",
      torch.allclose(m_iti._contrast_row(pos_c, neg_c), w_loose))
m_repe = RepEModel(RepEConfig(subject=SUBJECT))
check("RepEModel._contrast_row == repe_row", torch.allclose(m_repe._contrast_row(pos_X, neg_X), row))


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
