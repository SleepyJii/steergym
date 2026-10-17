#!/usr/bin/env python3
# Offline tests for the transform-steering chassis (steergym/transform.py)
# and the odesteer transform family (steergym/odesteer/{transforms,sketch}.py).
# NEEDS torch (CPU is enough). Run:
#   PYTHONPATH=<pylibs>:<torchlibs> python3 steergym/tests/test_transforms.py
#
# The suite is two parts:
# - a self-contained core: chassis position semantics on a fake layer,
#   closed-form fits vs hand-computed references, sketch finite-difference
#   oracles, persistence round-trips;
# - a source-parity section vs odesteer's own classes + sklearn/POT/
#   torchdiffeq, run only when ODESTEER_DIR points at a clone and those
#   libraries import (locally: fidlibs/t2libs on PYTHONPATH). It prints
#   SKIP and exits green without them — the core never silently shrinks.
#
# Model-level fits (extraction included) live in the devtools fake-LM smoke
# harness, not here.
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from steergym.generator import GeneratorRef
from steergym.transform import TransformSteerConfig, TransformSteerContext
from steergym.vector import VectorAdditionContext
from steergym.odesteer import (
    GatedGaussianTransportMap,
    LastTokenTransformConfig,
    LinActConfig,
    LinActModel,
    LinearTransportMap,
    MiMiCConfig,
    MiMiCModel,
    NormedPolyCntSketch,
    ODEMap,
    ODESteerConfig,
    ODESteerModel,
    PolyCntSketch,
    StepODEMap,
    StepODESteerConfig,
    StepODESteerModel,
    gaussian_ot_map,
    integrate_fixed_step,
    logistic_fit,
)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


SUBJECT = GeneratorRef("fake/subject")
torch.manual_seed(0)


# ---------------------------------------------------------------- config defaults (provenance)
ode = ODESteerConfig(subject=SUBJECT)
check("ODESteerConfig paper conf: degree=2 n_components=8000 gamma=0.1 coef0=1.0",
      (ode.degree, ode.n_components, ode.gamma, ode.coef0) == (2, 8000, 0.1, 1.0))
check("ODESteerConfig paper conf: solver=euler steps=10",
      (ode.solver, ode.steps) == ("euler", 10))
check("ODESteerConfig inherits extraction knobs (batch_size=6, max_length=1024, seed=42)",
      (ode.batch_size, ode.max_length, ode.seed) == (6, 1024, 42))
check("MiMiCConfig: ot_reg=1e-2 (their LinearTransport reg), C=1.0 gate",
      (MiMiCConfig(subject=SUBJECT).ot_reg, MiMiCConfig(subject=SUBJECT).inverse_reg_c)
      == (1e-2, 1.0))
check("config_cls wiring",
      (LinActModel.config_cls, MiMiCModel.config_cls,
       StepODESteerModel.config_cls, ODESteerModel.config_cls)
      == (LinActConfig, MiMiCConfig, StepODESteerConfig, ODESteerConfig))
check("weight stems: LinAcT / MiMiC / StepODESteer / ODESteer",
      (LinActModel.weight_name, MiMiCModel.weight_name,
       StepODESteerModel.weight_name, ODESteerModel.weight_name)
      == ("LinAcT", "MiMiC", "StepODESteer", "ODESteer"))
check("LastTokenTransformConfig is a TransformSteerConfig",
      isinstance(LastTokenTransformConfig(subject=SUBJECT), TransformSteerConfig))


# ---------------------------------------------------------------- chassis position semantics
class FakeLayer(nn.Module):
    def forward(self, h):
        return (h, "aux")  # tuple output like a decoder layer


class RowAddTransform(nn.Module):
    def __init__(self, row):
        super().__init__()
        self.register_buffer("row", row)

    def steer(self, X, T=1.0):
        return X + T * self.row


d = 8
row = torch.randn(d)
layer = FakeLayer()
prefill = torch.randn(2, 5, d)

ctx = TransformSteerContext(layer, RowAddTransform(row), T=2.0, positions="generated")
with ctx:
    out, aux = layer(prefill.clone())
check("generated: tuple output preserved", aux == "aux")
check("generated: prefill touches ONLY position -1",
      torch.allclose(out[:, -1, :], prefill[:, -1, :] + 2.0 * row)
      and torch.equal(out[:, :-1, :], prefill[:, :-1, :]))

decode = torch.randn(2, 1, d)  # KV-cache decode forward: len-1
with ctx:
    out2, _ = layer(decode.clone())
check("generated: len-1 decode forward steers its single (new-token) position",
      torch.allclose(out2[:, 0, :], decode[:, 0, :] + 2.0 * row))

diag = ctx.diagnostics()
check("diagnostics: rows accumulate across forwards (2 prefill + 2 decode)",
      diag["steered_rows"] == 4)
check("diagnostics: applied delta norm = T*||row|| for a row addition",
      abs(diag["applied_delta_norm_mean"] - 2.0 * float(row.norm())) < 1e-4
      and abs(diag["applied_delta_norm_max"] - 2.0 * float(row.norm())) < 1e-4)

h = torch.randn(3, 4, d)
with TransformSteerContext(layer, RowAddTransform(row), T=1.5, positions="all") as c_all:
    out_all, _ = layer(h.clone())
with VectorAdditionContext(layer, row, strength=1.5, positions="all"):
    out_vec, _ = layer(h.clone())
check("positions='all' row-addition == VectorAdditionContext (certified site)",
      torch.allclose(out_all, out_vec, atol=1e-6))

with VectorAdditionContext(layer, row, strength=1.5, positions="generated"):
    out_vg, _ = layer(h.clone())
check("VectorAdditionContext positions='generated' touches only -1",
      torch.allclose(out_vg[:, -1, :], h[:, -1, :] + 1.5 * row)
      and torch.equal(out_vg[:, :-1, :], h[:, :-1, :]))
check("hook removed on exit (no steering outside the context)",
      torch.equal(layer(h.clone())[0], h))

try:
    TransformSteerContext(layer, RowAddTransform(row), positions="last")
    check("invalid positions rejected", False)
except ValueError:
    check("invalid positions rejected", True)

# nested entry would register a second hook and leak it on exit
with TransformSteerContext(layer, RowAddTransform(row), T=1.0) as c_re:
    try:
        c_re.__enter__()
        check("nested context entry raises (hook-leak guard)", False)
    except RuntimeError:
        check("nested context entry raises (hook-leak guard)", True)
check("sequential re-entry still works after exit",
      torch.equal(layer(h.clone())[0], h))  # no hook left behind
v_re = VectorAdditionContext(layer, row, strength=1.0)
with v_re:
    try:
        v_re.__enter__()
        check("VectorAdditionContext nested entry raises too", False)
    except RuntimeError:
        check("VectorAdditionContext nested entry raises too", True)

# a non-affine transform through the chassis: result dtype follows the stream
class HalveTransform(nn.Module):
    def steer(self, X, T=1.0):
        return 0.5 * X


h16 = torch.randn(1, 3, d, dtype=torch.bfloat16)
with TransformSteerContext(layer, HalveTransform(), positions="generated"):
    outh, _ = layer(h16.clone())
check("stream dtype preserved (transform runs float32, result cast back)",
      outh.dtype == torch.bfloat16
      and torch.allclose(outh[:, -1, :].float(), 0.5 * h16[:, -1, :].float(), atol=1e-2))


# ---------------------------------------------------------------- LinOT closed form
g = torch.Generator().manual_seed(1)
pos_X = torch.randn(40, d, generator=g) * 1.7 + 0.5
neg_X = torch.randn(55, d, generator=g)

lin = LinearTransportMap().fit(target_X=pos_X, source_X=neg_X)
n = min(len(pos_X), len(neg_X))
t, s = pos_X[:n], neg_X[:n]
m_t, m_s = t.mean(0), s.mean(0)
t_c, s_c = (t - m_t).sort(dim=0).values, (s - m_s).sort(dim=0).values
w_ref = (t_c * s_c).sum(0) / ((s_c ** 2).sum(0) + 1e-10)
check("LinOT: w matches the hand-computed sorted-quantile formula",
      torch.equal(lin.w, w_ref))
check("LinOT: b = m_target - w*m_source", torch.equal(lin.b, m_t - w_ref * m_s))
X = torch.randn(7, d, generator=g)
check("LinOT: steer = w (.) X + b, T ignored (their LinAcT.steer)",
      torch.equal(lin.steer(X, T=5.0), X * lin.w + lin.b)
      and torch.equal(lin.steer(X, T=5.0), lin.steer(X, T=0.0)))
lin_trunc = LinearTransportMap().fit(target_X=pos_X, source_X=neg_X[:len(pos_X)])
check("LinOT: unequal set sizes truncate-pair (tail of larger set ignored)",
      torch.equal(lin.w, lin_trunc.w) and torch.equal(lin.b, lin_trunc.b))


# ---------------------------------------------------------------- Gaussian OT closed form
# d=1 analytic case: A = sqrt((var_t + reg)/(var_s + reg)), biased variances
src = torch.randn(400, 1, generator=g) * 2.0
tgt = torch.randn(300, 1, generator=g) * 0.5 + 3.0
A1, b1 = gaussian_ot_map(src, tgt, reg=1e-2)
var_s = src.double().var(dim=0, unbiased=False).item() + 1e-2
var_t = tgt.double().var(dim=0, unbiased=False).item() + 1e-2
A_ref = (var_t / var_s) ** 0.5
check("gaussian_ot_map d=1: A = sqrt((var_t+reg)/(var_s+reg)) with BIASED variances",
      abs(A1.item() - A_ref) < 1e-6)
check("gaussian_ot_map d=1: b = m_t - A*m_s",
      abs(b1.item() - (tgt.double().mean() - A_ref * src.double().mean()).item()) < 1e-5)

# multivariate: A symmetric and A Sigma_s A ~= Sigma_t (the transport property)
srcm = torch.randn(600, 4, generator=g) @ torch.randn(4, 4, generator=g)
tgtm = torch.randn(500, 4, generator=g) @ torch.randn(4, 4, generator=g) + 1.0
Am, bm = gaussian_ot_map(srcm, tgtm, reg=1e-2)
Cs = ((srcm - srcm.mean(0)).T @ (srcm - srcm.mean(0)) / len(srcm)
      + 1e-2 * torch.eye(4)).double()
Ct = ((tgtm - tgtm.mean(0)).T @ (tgtm - tgtm.mean(0)) / len(tgtm)
      + 1e-2 * torch.eye(4)).double()
check("gaussian_ot_map: A symmetric", torch.allclose(Am, Am.T, atol=1e-5))
check("gaussian_ot_map: A Sigma_s A == Sigma_t (transport property)",
      torch.allclose(Am.double() @ Cs @ Am.double(), Ct, rtol=1e-3, atol=1e-4))


# ---------------------------------------------------------------- MiMiC gate semantics
mimic = GatedGaussianTransportMap().fit(pos_X + 3.0, neg_X - 3.0)  # separable
probe = torch.cat([pos_X[:5] + 3.0, neg_X[:5] - 3.0])
steered = mimic.steer(probe)
decision = probe @ mimic.gate_w + mimic.gate_b
check("MiMiC: rows predicted positive (decision>0) pass through untouched",
      torch.equal(steered[decision > 0], probe[decision > 0]))
check("MiMiC: rows predicted negative get A h + b, T ignored",
      torch.allclose(steered[decision <= 0],
                     probe[decision <= 0] @ mimic.A + mimic.b)
      and torch.equal(mimic.steer(probe, T=9.0), steered))
check("MiMiC: separable sets gate cleanly (all pos kept, all neg moved)",
      bool((decision[:5] > 0).all()) and bool((decision[5:] <= 0).all()))
# exact decision tie: sklearn predict maps decision==0 to class 0 -> transported
tie = GatedGaussianTransportMap()
tie.register_buffer("gate_w", torch.zeros(d).index_fill_(0, torch.tensor(0), 1.0))
tie.register_buffer("gate_b", torch.tensor(0.0))
tie.register_buffer("A", 2.0 * torch.eye(d))
tie.register_buffer("b", torch.zeros(d))
x_tie = torch.zeros(1, d)  # decision exactly 0.0
check("MiMiC: a row at decision exactly 0 is transported (sklearn tie semantics)",
      torch.equal(tie.steer(x_tie), x_tie @ tie.A + tie.b))

moved = mimic.steer(neg_X - 3.0)
check("MiMiC: transported negatives land near the positive mean",
      float((moved.mean(0) - (pos_X + 3.0).mean(0)).norm())
      < 0.2 * float(((neg_X - 3.0).mean(0) - (pos_X + 3.0).mean(0)).norm()))


# ---------------------------------------------------------------- sketch: kernel + oracles
D_FEAT, NC = 16, 2048
gen = torch.Generator().manual_seed(7)
sk = PolyCntSketch(degree=2, n_components=NC, gamma=0.5, coef0=1.0)
xk = torch.randn(30, D_FEAT, generator=gen, dtype=torch.float64)
sk.fit(xk, generator=torch.Generator().manual_seed(11))
# y = x keeps the kernel value large relative to TensorSketch's additive
# error ~ ||x||^d ||y||^d / sqrt(n_components) (uncorrelated pairs drown)
Zx = sk.transform(xk)
kern_hat = (Zx * Zx).sum(dim=1)
kern_true = (0.5 * (xk * xk).sum(dim=1) + 1.0) ** 2
rel = ((kern_hat - kern_true).abs() / kern_true).median()
check(f"PolyCntSketch: z(x).z(x) approximates (gamma x.x + c0)^2 (median rel err {rel:.3f})",
      float(rel) < 0.1)

check("sketch fit is generator-deterministic",
      torch.equal(sk.indexHash_,
                  PolyCntSketch(2, NC, 0.5, 1.0).fit(
                      xk, generator=torch.Generator().manual_seed(11)).indexHash_))

# finite-difference oracle for the analytical vjp (float64)
def fd_vjp(module, x, v, eps=1e-6):
    out = torch.zeros_like(x)
    for j in range(len(x)):
        xp, xm = x.clone(), x.clone()
        xp[j] += eps
        xm[j] -= eps
        out[j] = ((module.transform(xp) - module.transform(xm)) @ v) / (2 * eps)
    return out


x1 = torch.randn(D_FEAT, generator=gen, dtype=torch.float64)
v1 = torch.randn(NC, generator=gen, dtype=torch.float64)
check("PolyCntSketch: analytical vjp matches finite differences",
      torch.allclose(sk.vjp(x1, v1), fd_vjp(sk, x1, v1), rtol=1e-4, atol=1e-6))
check("PolyCntSketch: vjp == grad(X)^T v (two code paths agree)",
      torch.allclose(sk.vjp(x1, v1), sk.grad(x1).T @ v1, rtol=1e-6, atol=1e-8))

nsk = NormedPolyCntSketch(degree=2, n_components=NC, gamma=0.5, coef0=1.0)
nsk.fit(xk, generator=torch.Generator().manual_seed(11))
check("NormedPolyCntSketch: scale-invariant forward (z(x) == z(10x))",
      torch.allclose(nsk.transform(x1), nsk.transform(10.0 * x1), atol=1e-8))
check("NormedPolyCntSketch: vjp matches finite differences (normalization chain rule)",
      torch.allclose(nsk.vjp(x1, v1), fd_vjp(nsk, x1, v1), rtol=1e-4, atol=1e-6))
check("NormedPolyCntSketch: vjp orthogonal to x (radial directions are dead)",
      abs(float(nsk.vjp(x1, v1) @ x1)) < 1e-6 * float(x1.norm()) * float(nsk.vjp(x1, v1).norm() + 1e-12))
check("NormedPolyCntSketch: batched vjp == per-row vjp (upstream's missing "
      "sqrt(gamma) in the batch path is fixed — see sketch.py docstring)",
      torch.allclose(nsk.vjp(xk[:4], v1), torch.stack([nsk.vjp(r, v1) for r in xk[:4]]),
                     rtol=1e-6, atol=1e-8))
check("PolyCntSketch: batched vjp matches finite differences too",
      torch.allclose(sk.vjp(x1.unsqueeze(0), v1).squeeze(0), fd_vjp(sk, x1, v1),
                     rtol=1e-4, atol=1e-6))


# ---------------------------------------------------------------- kernel logistic + ODE maps
# small, separable synthetic concept
gen2 = torch.Generator().manual_seed(3)
direction = torch.randn(D_FEAT, generator=gen2)
base = torch.randn(120, D_FEAT, generator=gen2)
posf = (base + 1.5 * direction).float()
negf = (base - 1.5 * direction + 0.3 * torch.randn(120, D_FEAT, generator=gen2)).float()

step = StepODEMap(degree=2, n_components=512, gamma=0.1, coef0=1.0)
step.fit(posf, negf, generator=torch.Generator().manual_seed(5))
check("StepODEMap: dre_coeff = (1-pi)/pi (balanced sets -> 1.0)",
      abs(float(step.dre_coeff) - 1.0) < 1e-6)
probe_f = negf[:6]
vf = step.vector_field(probe_f)
check("vector_field is unit-norm per row",
      torch.allclose(vf.norm(dim=-1), torch.ones(6), atol=1e-4))
check("vector_field points toward the positive class on negatives",
      float((vf * (posf.mean(0) - probe_f)).sum(dim=-1).gt(0).float().mean()) > 0.9)
check("StepODEMap: steer = X + T*v(X) exactly",
      torch.allclose(step.steer(probe_f, T=3.0), probe_f + 3.0 * vf, atol=1e-6))

odem = ODEMap(degree=2, n_components=512, gamma=0.1, coef0=1.0, solver="euler", steps=10)
odem.fit(posf, negf, generator=torch.Generator().manual_seed(5))
check("ODEMap: identical seed -> identical sketch + coef as StepODEMap",
      torch.equal(odem.kernel.indexHash_, step.kernel.indexHash_)
      and torch.allclose(odem.coef, step.coef, atol=1e-6))

X0 = probe_f.clone()
Xm = probe_f.clone()
for _ in range(10):
    Xm = Xm + (3.0 / 10) * odem.vector_field(Xm)
check("ODEMap euler: 10 native steps == 10 manual explicit Euler steps",
      torch.allclose(odem.steer(probe_f, T=3.0), Xm, atol=1e-6))
check("ODEMap: T=0 returns X unchanged (their short-circuit)",
      torch.equal(odem.steer(probe_f, T=0.0), probe_f))
odem1 = ODEMap(degree=2, n_components=512, gamma=0.1, coef0=1.0, solver="euler", steps=1)
odem1.fit(posf, negf, generator=torch.Generator().manual_seed(5))
check("ODEMap euler steps=1 == StepODEMap at the same T",
      torch.allclose(odem1.steer(probe_f, T=2.0), step.steer(probe_f, T=2.0), atol=1e-6))
try:
    integrate_fixed_step(lambda x: x, X0, 1.0, "dopri5", 2)
    check("unknown solver raises", False)
except ValueError:
    check("unknown solver raises", True)
try:
    integrate_fixed_step(lambda x: x, X0, 1.0, "euler", 0)
    check("steps < 1 raises", False)
except ValueError:
    check("steps < 1 raises", True)
check("repr of an unfitted sketch does not raise",
      "unfitted" in repr(NormedPolyCntSketch(2, 64, 0.1, 1.0)))

# ODE moves barrier value up: classifier logit increases along the flow
logit0 = step.kernel.transform(probe_f) @ step.coef + step.intercept
logit1 = step.kernel.transform(odem.steer(probe_f, T=3.0)) @ step.coef + step.intercept
check("ODE flow increases the classifier logit (forward-invariance direction)",
      bool((logit1 > logit0).all()))


# ---------------------------------------------------------------- persistence round-trips
def _mk(model_cls, cfg):
    m = model_cls(cfg)
    m.transforms["concept-a"] = m._fit_transform(posf[:, :D_FEAT], negf[:, :D_FEAT])
    return m


tmp = Path(tempfile.mkdtemp(prefix="transforms-test-"))
small_ode_cfg = ODESteerConfig(subject=SUBJECT, n_components=512, seed=42)
for model_cls, cfg in [
    (LinActModel, LinActConfig(subject=SUBJECT)),
    (MiMiCModel, MiMiCConfig(subject=SUBJECT)),
    (StepODESteerModel, StepODESteerConfig(subject=SUBJECT, n_components=512)),
    (ODESteerModel, small_ode_cfg),
]:
    m = _mk(model_cls, cfg)
    p = tmp / f"{model_cls.weight_name}.pt"
    m.save(p)
    m2 = model_cls.load(p)
    a = m.transforms["concept-a"].steer(probe_f, T=2.0)
    b = m2.transforms["concept-a"].steer(probe_f, T=2.0)
    check(f"{model_cls.weight_name}: save/load round-trip steers bit-equal",
          torch.equal(a, b))
    check(f"{model_cls.weight_name}: loaded config round-trips",
          m2.config == cfg)

m_a = _mk(ODESteerModel, small_ode_cfg)
m_b = _mk(ODESteerModel, small_ode_cfg)
check("ODESteer: same config.seed -> identical artifacts (hashes + coef)",
      torch.equal(m_a.transforms["concept-a"].kernel.indexHash_,
                  m_b.transforms["concept-a"].kernel.indexHash_)
      and torch.allclose(m_a.transforms["concept-a"].coef,
                         m_b.transforms["concept-a"].coef, atol=1e-7))

# load() strictness: wrong class, wrong buffer set, config/state mismatch
try:
    MiMiCModel.load(tmp / "LinAcT.pt")
    check("load: wrong-class artifact rejected at load time", False)
except ValueError:
    check("load: wrong-class artifact rejected at load time", True)
_d = torch.load(tmp / "MiMiC.pt", weights_only=False)
_d["cls"] = "LinActModel"
for extra in ("inverse_reg_c", "ot_reg"):  # keep the config parseable as LinActConfig
    _d["config"].pop(extra)
torch.save(_d, tmp / "mislabeled.pt")
try:
    LinActModel.load(tmp / "mislabeled.pt")
    check("load: buffer-set mismatch rejected (MiMiC state under LinAcT cls)", False)
except ValueError:
    check("load: buffer-set mismatch rejected (MiMiC state under LinAcT cls)", True)
_d2 = torch.load(tmp / "ODESteer.pt", weights_only=False)
_d2["config"]["n_components"] = 256  # artifact coef stays 512-dim
torch.save(_d2, tmp / "tampered.pt")
try:
    ODESteerModel.load(tmp / "tampered.pt")
    check("load: sketch state disagreeing with config rejected", False)
except ValueError:
    check("load: sketch state disagreeing with config rejected", True)


# ================================================================ source parity (optional)
ODESTEER_DIR = os.environ.get("ODESTEER_DIR", "")
have_libs = all(importlib.util.find_spec(m) for m in ("sklearn", "ot", "torchdiffeq"))
if not (ODESTEER_DIR and Path(ODESTEER_DIR).is_dir() and have_libs):
    print("SKIP source-parity section (needs ODESTEER_DIR + sklearn/ot/torchdiffeq)")
else:
    sys.path.insert(0, str(Path(ODESTEER_DIR) / "src"))
    from odesteer.steer._lin_act import LinAcT as TheirLinAcT
    from odesteer.steer._mimic import MiMiC as TheirMiMiC
    from odesteer.steer._ode_steer import ODESteer as TheirODESteer
    from odesteer.utils.kernels import NormedPolyCntSketch as TheirSketch

    theirs_lin = TheirLinAcT().fit(pos_X, neg_X)
    check("parity LinAcT: steer bit-equal to their class",
          torch.equal(lin.steer(X), theirs_lin.steer(X)))

    # MiMiC: formula parity at float64 (pins POT's reg convention), gate
    # agreement at the decision level (solvers differ in tolerance only)
    import numpy as np
    import ot as potlib

    lt = potlib.da.LinearTransport(reg=1e-2)
    lt.fit(Xs=(negf.double().numpy()), Xt=(posf.double().numpy()))
    A_ours, b_ours = gaussian_ot_map(negf, posf, reg=1e-2)
    check("parity MiMiC/OT: A matches POT LinearTransport (float64 formula)",
          torch.allclose(A_ours.double(), torch.from_numpy(np.asarray(lt.A_)),
                         rtol=1e-4, atol=1e-6))
    check("parity MiMiC/OT: b matches POT",
          torch.allclose(b_ours.double(),
                         torch.from_numpy(np.asarray(lt.B_)).reshape(-1),
                         rtol=1e-4, atol=1e-5))

    theirs_mimic = TheirMiMiC().fit(posf, negf)
    ours_mimic = GatedGaussianTransportMap().fit(posf, negf)
    probe_m = torch.cat([posf[:20], negf[:20]])
    their_pred = torch.as_tensor(theirs_mimic.clf.predict(probe_m.numpy()))
    our_pred = (probe_m @ ours_mimic.gate_w + ours_mimic.gate_b) > 0
    agree = float((their_pred.bool() == our_pred).float().mean())
    check(f"parity MiMiC: gate decisions agree with sklearn ({agree:.3f})", agree >= 0.95)
    their_out = theirs_mimic.steer(probe_m)
    our_out = ours_mimic.steer(probe_m)
    check("parity MiMiC: steered outputs close to their full fp32 pipeline",
          torch.allclose(their_out, our_out, rtol=1e-2, atol=1e-2))

    # sketch A/B: inject their realized buffers into our module -> bit-equal
    torch.manual_seed(123)
    theirs_sk = TheirSketch(degree=2, n_components=256, gamma=0.1, coef0=1.0)
    theirs_sk.fit(posf)
    ours_sk = NormedPolyCntSketch(degree=2, n_components=256, gamma=0.1, coef0=1.0)
    ours_sk.register_buffer("nFeat_", torch.tensor(theirs_sk.n_features_))
    ours_sk.register_buffer("indexHash_", theirs_sk.indexHash_.clone())
    ours_sk.register_buffer("bitHash_", theirs_sk.bitHash_.clone())
    check("parity sketch: transform bit-equal under injected buffers",
          torch.equal(ours_sk.transform(posf[:8]), theirs_sk.transform(posf[:8])))
    v256 = torch.randn(256)
    check("parity sketch: single-row vjp bit-equal under injected buffers",
          torch.equal(ours_sk.vjp(posf[2], v256), theirs_sk.vjp(posf[2], v256)))
    # upstream's batch vjp omits the sqrt(gamma) factor (inert there: the
    # vector field unit-normalizes); ours applies it — exact scalar relation
    check("parity sketch: batch vjp == sqrt(gamma) * theirs (their batch-path bug)",
          torch.allclose(ours_sk.vjp(posf[:8], v256),
                         (0.1 ** 0.5) * theirs_sk.vjp(posf[:8], v256),
                         rtol=1e-6, atol=1e-7))
    check("parity sketch: grad bit-equal under injected buffers",
          torch.equal(ours_sk.grad(posf[2]), theirs_sk.grad(posf[2])))

    # full classifier fit parity: their sklearn-on-Z vs our LBFGS-on-Z,
    # same injected features
    torch.manual_seed(321)
    theirs_ode = TheirODESteer(solver="euler", steps=10, n_components=512,
                               degree=2, gamma=0.1, coef0=1.0, lin_clf_type="lr")
    theirs_ode.fit(posf, negf)
    ours_ode = ODEMap(degree=2, n_components=512, gamma=0.1, coef0=1.0,
                      solver="euler", steps=10)
    # inject their sketch so the only difference is the logistic solver
    ours_ode.kernel.register_buffer("nFeat_", torch.tensor(theirs_ode.clf.kernel.n_features_))
    ours_ode.kernel.register_buffer("indexHash_", theirs_ode.clf.kernel.indexHash_.clone())
    ours_ode.kernel.register_buffer("bitHash_", theirs_ode.clf.kernel.bitHash_.clone())
    coef, intercept = logistic_fit(ours_ode.kernel.transform(posf),
                                   ours_ode.kernel.transform(negf), 1.0)
    ours_ode.register_buffer("coef", coef)
    ours_ode.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))
    ours_ode.register_buffer("dre_coeff", torch.tensor(1.0))

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(
            a.double().flatten(), b.double().flatten(), 0))

    c_coef = cos(ours_ode.coef, theirs_ode.clf.coef)
    check(f"parity clf: coef cosine vs sklearn fit > 0.99 ({c_coef:.5f})", c_coef > 0.99)
    vf_ours = ours_ode.vector_field(probe_f)
    vf_theirs = theirs_ode.vector_field(probe_f)
    c_vf = min(cos(a, b) for a, b in zip(vf_ours, vf_theirs))
    check(f"parity clf: per-row vector-field cosine > 0.99 (min {c_vf:.5f})", c_vf > 0.99)

    # native integrator vs torchdiffeq on the SAME field (theirs)
    from torchdiffeq import odeint

    for solver in ("euler", "midpoint", "rk4"):
        ref = odeint(lambda t, s: theirs_ode.vector_field(s), probe_f,
                     torch.tensor([0., 3.0]), method=solver,
                     options={"step_size": 3.0 / 10})[1]
        nat = integrate_fixed_step(theirs_ode.vector_field, probe_f, 3.0, solver, 10)
        err = float((ref - nat).abs().max())
        check(f"parity integrator[{solver}]: native == torchdiffeq (max {err:.2e})",
              err < 1e-4)

    # NONLINEAR field pins the tableaus (the fitted near-unit field is too
    # locally constant, and any 4-stage 4th-order method agrees on a linear
    # field): vf = sin(x) elementwise, steps=1
    vf_nl = lambda x: torch.sin(x)
    Xc = torch.randn(6, 4, generator=torch.Generator().manual_seed(9)) * 2.0
    for solver in ("euler", "midpoint", "rk4"):
        ref = odeint(lambda t, s: vf_nl(s), Xc, torch.tensor([0., 1.0]),
                     method=solver, options={"step_size": 1.0})[1]
        nat = integrate_fixed_step(vf_nl, Xc, 1.0, solver, 1)
        err = float((ref - nat).abs().max())
        check(f"parity integrator[{solver}] on a NONLINEAR field, steps=1 (max {err:.2e})",
              err < 1e-6)
    # and the textbook rk4 tableau would NOT pass that check
    k1 = vf_nl(Xc); k2 = vf_nl(Xc + 0.5 * k1); k3 = vf_nl(Xc + 0.5 * k2); k4 = vf_nl(Xc + k3)
    classic = Xc + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    ref_rk4 = odeint(lambda t, s: vf_nl(s), Xc, torch.tensor([0., 1.0]),
                     method="rk4", options={"step_size": 1.0})[1]
    check("nonlinear-field check is discriminating (classic RK4 differs from torchdiffeq's 3/8)",
          float((classic - ref_rk4).abs().max()) > 1e-4)

    # end-to-end: their ODESteer.steer vs ours with identical clf state
    their_steered = theirs_ode.steer(probe_f, T=3.0)
    ours_ode.coef = theirs_ode.clf.coef.clone()          # buffer assignment
    ours_ode.intercept = theirs_ode.clf.intercept.clone().reshape(())
    ours_steered = ours_ode.steer(probe_f, T=3.0)
    err = float((their_steered - ours_steered).abs().max())
    check(f"parity ODESteer: end-to-end steer matches theirs under shared clf (max {err:.2e})",
          err < 1e-4)


# ---------------------------------------------------------------- summary
print(f"\n{'PASS' if not FAILURES else 'FAIL'}: "
      f"{len(FAILURES)} failures" + (f" -> {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
