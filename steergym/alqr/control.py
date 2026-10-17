"""A-LQR's finite-horizon Riccati recursion (upstream steer/lqr_utils.py's
time_varying_lqr_noB_mem_efficient), isotropic costs, B = I:

    S_L = q_f * I
    for t = L-1 .. 0:
        P   = S_{t+1} + r * I                               (B^T S B + R)
        F   = S_{t+1} @ A_t                                 (B^T S A)
        K_t = P^{-1} @ F
        S_t = q * I + A_t^T @ S_{t+1} @ A_t  -  F^T @ P^{-1} @ F

riccati_gains returns every K_t [n, n]; riccati_gain_rows returns only
w_t = K_t @ V[t] -- all the fitted artifact ever stores. Op order matches
upstream bit for bit (see the solve comments below). Torch imported lazily:
`import steergym` must stay usable in a bare venv."""


def riccati_gains(A, q: float, r: float, q_f: float):
    import torch

    L, n, _ = A.shape
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    K = torch.zeros(L, n, n, dtype=A.dtype, device=A.device)
    S = q_f * eye
    for t in reversed(range(L)):
        At = A[t]
        P = S + r * eye
        F = S @ At
        G = q * eye + At.transpose(-2, -1) @ S @ At
        K[t] = torch.linalg.solve(P, F)
        # upstream solves (P, F) twice (K_t, then again in the S update)
        # instead of reusing K_t -- kept for bit-parity
        S = G - F.transpose(-2, -1) @ torch.linalg.solve(P, F)
    return K


def riccati_gain_rows(A, V, q: float, r: float, q_f: float):
    import torch

    L, n, _ = A.shape
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    W = torch.zeros(L, n, dtype=A.dtype, device=A.device)
    S = q_f * eye
    for t in reversed(range(L)):
        At = A[t]
        P = S + r * eye
        F = S @ At
        G = q * eye + At.transpose(-2, -1) @ S @ At
        # solve() returns non-C-contiguous; riccati_gains's K[t] = ... copies
        # to C-contiguous, and BLAS kernels differ by layout -- .contiguous()
        # keeps the two functions bitwise equal
        K_t = torch.linalg.solve(P, F).contiguous()
        W[t] = K_t @ V[t]
        S = G - F.transpose(-2, -1) @ torch.linalg.solve(P, F)
    return W
