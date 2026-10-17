"""odesteer's transform family — methods whose artifact is a map h -> f(h):
LinAcT (Apple ml-act's linear optimal transport), MiMiC (classifier-gated
Gaussian optimal transport), StepODESteer / ODESteer (the paper's method:
one Euler step / fixed-step ODE integration along the normalized gradient
of a kernel-logistic barrier classifier over polynomial count-sketch
features — see sketch.py's docstring for how the sketch works).

This module holds the PURE transforms — fit/steer torch Modules with no
steergym model plumbing. Every transform exposes odesteer's calling
convention `.steer(X [N, d], T) -> [N, d]`, so a fitted object drops
straight into their harness (`model.steer_model`) for oracle
certification. The per-concept SteerModel wrappers (configs, extraction,
persistence) live in transform_models.py; steering in OUR runtime rides
TransformSteerContext (steergym/transform.py).

Deltas vs odesteer (all deliberate, mirrors the vector pack's):
- fits are torch-native (LBFGS logistic, eigh matrix-sqrt) — no sklearn,
  POT, or torchdiffeq dependency; each objective is convex/closed-form so
  the optimum is solver-independent (source-parity suite-checked).
- ODESteer integrates with a native fixed-step solver reproducing
  torchdiffeq's fixed-grid methods (euler; midpoint; rk4 = torchdiffeq's
  3/8-rule variant) — parity vs torchdiffeq lives in the test suite only.
- the count-sketch hash draw is seeded from config.seed (upstream draws
  from the ambient global RNG), and the realized hash buffers are saved in
  the artifact.
"""
import torch
from torch import Tensor, nn

from .models import logistic_fit
from .sketch import NormedPolyCntSketch


# ---------------------------------------------------------------- transforms
class LinearTransportMap(nn.Module):
    """odesteer _lin_act.py's LinOT ("linear optimal transport"), verbatim:
    the coordinate-wise affine map transporting the source (negative)
    activation distribution onto the target (positive) one — per-coordinate
    sorting pairs the two empirical quantile functions, and w/b are the
    least-squares line through those paired quantiles: truncate-pair,
    mean-center, sort both sets per coordinate, w = sum(t*s)/sum(s^2 + eps),
    b = m_target - w * m_source; f(h) = w (.) h + b. T is unused (their
    LinAcT.steer ignores it; conf T=1)."""

    def fit(self, target_X: Tensor, source_X: Tensor):
        n = min(len(target_X), len(source_X))
        target_X, source_X = target_X[:n], source_X[:n]
        m_target, m_source = target_X.mean(dim=0), source_X.mean(dim=0)
        target_centered = (target_X - m_target).sort(dim=0).values
        source_centered = (source_X - m_source).sort(dim=0).values
        w_num = torch.sum(target_centered * source_centered, dim=0)
        w_den = torch.sum(source_centered ** 2, dim=0)
        self.register_buffer("w", w_num / (w_den + 1e-10))
        self.register_buffer("b", m_target - self.w * m_source)
        return self

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        return X * self.w + self.b


def _sqrtm_psd(C: Tensor) -> Tensor:
    # symmetric PSD matrix square root via eigh (POT's nx.sqrtm up to fp)
    evals, evecs = torch.linalg.eigh(C)
    return (evecs * evals.clamp_min(0).sqrt()) @ evecs.T


def gaussian_ot_map(source_X: Tensor, target_X: Tensor, reg: float = 1e-2):
    """(A [d, d], b [d]) of the Gaussian optimal-transport closed form —
    the affine map that transports N(m_s, Cs) onto N(m_t, Ct) — matching POT's
    empirical_bures_wasserstein_mapping exactly: biased covariances
    (X_c^T X_c / n) with reg*I added to BOTH, A = Cs^(-1/2)
    (Cs^(1/2) Ct Cs^(1/2))^(1/2) Cs^(-1/2), b = m_t - A m_s. float64
    internally, float32 out."""
    Xs, Xt = source_X.double(), target_X.double()
    ms, mt = Xs.mean(dim=0), Xt.mean(dim=0)
    Xs_c, Xt_c = Xs - ms, Xt - mt
    eye = torch.eye(Xs.shape[1], dtype=torch.float64, device=Xs.device)
    Cs = Xs_c.T @ Xs_c / len(Xs) + reg * eye
    Ct = Xt_c.T @ Xt_c / len(Xt) + reg * eye
    Cs12 = _sqrtm_psd(Cs)
    Cs12inv = torch.linalg.inv(Cs12)
    M0 = _sqrtm_psd(Cs12 @ Ct @ Cs12)
    A = Cs12inv @ M0 @ Cs12inv
    b = mt - ms @ A  # A is symmetric, so this is POT's mt - dot(ms, A)
    return A.float(), b.float()


class GatedGaussianTransportMap(nn.Module):
    """odesteer _mimic.py MiMiC: a logistic gate (pos=1/neg=0, sklearn
    default objective) decides which rows are transported; rows predicted
    NEGATIVE get the Gaussian optimal-transport map neg->pos, A h + b.
    T is unused (their MiMiC.steer ignores it; conf T=1)."""

    def fit(self, pos_X: Tensor, neg_X: Tensor,
            inverse_reg_c: float = 1.0, ot_reg: float = 1e-2):
        gate_w, gate_b = logistic_fit(pos_X, neg_X, inverse_reg_c)
        self.register_buffer("gate_w", gate_w)
        self.register_buffer("gate_b", torch.tensor(gate_b, dtype=torch.float32))
        A, b = gaussian_ot_map(source_X=neg_X, target_X=pos_X, reg=ot_reg)
        self.register_buffer("A", A)
        self.register_buffer("b", b)
        return self

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        # sklearn predict: class 1 iff decision > 0 — rows at <= 0 are
        # predicted negative and get transported (their y_pred == 0 mask)
        neg = (X @ self.gate_w + self.gate_b) <= 0
        out = X.clone()
        out[neg] = X[neg] @ self.A + self.b  # A symmetric = POT's x @ A_ + B_
        return out


class StepODEMap(nn.Module):
    """odesteer _step_ode_steer.py StepODESteer: f(h) = h + T * v(h) with
    v = the unit-normalized gradient of a kernel-logistic density-ratio
    classifier over NormedPolyCntSketch features (their NormedPolyClassifier:
    dre_coeff = (1-pi)/pi, grad = kernel.vjp(X, coef) * dre_coeff)."""

    def __init__(self, degree: int = 2, n_components: int = 8000,
                 gamma: float = 0.1, coef0: float = 1.0):
        super().__init__()
        self.kernel = NormedPolyCntSketch(degree=degree, n_components=n_components,
                                          gamma=gamma, coef0=coef0)

    def fit(self, pos_X: Tensor, neg_X: Tensor, inverse_reg_c: float = 1.0,
            generator: torch.Generator | None = None):
        pi = len(pos_X) / (len(pos_X) + len(neg_X))
        self.register_buffer("dre_coeff",
                             torch.tensor((1 - pi) / pi, dtype=torch.float32))
        self.kernel.fit(torch.cat([pos_X, neg_X], dim=0), generator=generator)
        coef, intercept = logistic_fit(self.kernel.transform(pos_X),
                                       self.kernel.transform(neg_X), inverse_reg_c)
        self.register_buffer("coef", coef)
        self.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))
        return self

    def grad(self, X: Tensor) -> Tensor:
        return self.kernel.vjp(X, self.coef) * self.dre_coeff

    def vector_field(self, X: Tensor) -> Tensor:
        raw_grad = self.grad(X)
        return raw_grad / (raw_grad.norm(dim=-1, keepdim=True) + 1e-10)

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        return X + T * self.vector_field(X)


def integrate_fixed_step(vf, X: Tensor, T: float, solver: str, steps: int) -> Tensor:
    """Fixed-grid ODE integration of dX/dt = vf(X) from 0 to T in `steps`
    steps — the native replacement for torchdiffeq's fixed-grid solvers
    (parity-tested against odeint): euler; midpoint (explicit midpoint);
    rk4 = torchdiffeq's rk4_alt_step_func, the 3/8 rule."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    dt = T / steps
    for _ in range(steps):
        if solver == "euler":
            X = X + dt * vf(X)
        elif solver == "midpoint":
            X = X + dt * vf(X + 0.5 * dt * vf(X))
        elif solver == "rk4":
            k1 = vf(X)
            k2 = vf(X + dt * k1 / 3)
            k3 = vf(X + dt * (k2 - k1 / 3))
            k4 = vf(X + dt * (k1 - k2 + k3))
            X = X + dt * (k1 + 3 * (k2 + k3) + k4) / 8
        else:
            raise ValueError(f"solver {solver!r}; expected euler|midpoint|rk4")
    return X


class ODEMap(StepODEMap):
    """odesteer _ode_steer.py ODESteer: integrate the same vector field for
    time T (paper: euler, 10 steps => step size T/10)."""

    def __init__(self, degree: int = 2, n_components: int = 8000,
                 gamma: float = 0.1, coef0: float = 1.0,
                 solver: str = "euler", steps: int = 10):
        super().__init__(degree=degree, n_components=n_components,
                         gamma=gamma, coef0=coef0)
        self.solver = solver
        self.steps = int(steps)

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if T == 0.:
            return X
        return integrate_fixed_step(self.vector_field, X, T, self.solver, self.steps)
