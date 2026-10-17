#!/usr/bin/env python3
# Offline tests for steergym.alqr (A-LQR): the Riccati recursion
# (control.py), the per-layer Jacobian extraction (linearize.py), the
# ALQRContext closed-loop hook math, and ALQRModel fit/save/load. Needs
# torch (SVD-free but autograd.functional.jacobian / linalg.solve) -- CPU
# torch is enough. Run:
#   PYTHONPATH=<pylibs>:<torchlibs> python3 steergym/tests/test_alqr.py
# An optional ALQR_DIR-gated leg replays upstream's own Riccati recursion
# (and, best-effort, its runtime hook) from the vendored source repo.
import os
import random
import sys
import types
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

# Resolved now, before section 4 below replaces sys.modules["transformers"]
# with a fake-LM-only stub (it only defines set_seed): the real-HF leg
# (section 7) needs the genuine package's model/config classes regardless of
# that later stubbing. A bare `import transformers` captured as a module
# reference is NOT enough -- transformers is a lazy-attribute module whose
# `.LlamaConfig` etc. resolve via `importlib.import_module(..., self.__name__)`
# at ATTRIBUTE-ACCESS time, which looks `self.__name__` ("transformers") up
# in sys.modules; accessed after the stub is installed this raises
# "ModuleNotFoundError: 'transformers' is not a package". So the classes
# are pulled out and cached here, while sys.modules still holds the real
# package.
try:
    import transformers as _real_transformers
    _RealLlamaConfig = _real_transformers.LlamaConfig
    _RealLlamaForCausalLM = _real_transformers.LlamaForCausalLM
    _RealQwen2Config = _real_transformers.Qwen2Config
    _RealQwen2ForCausalLM = _real_transformers.Qwen2ForCausalLM
except ImportError:
    _real_transformers = None

from steergym.concept import Concept
from steergym.generator import GeneratorLM, GeneratorRef
from steergym.alqr import ALQRConfig, ALQRContext, ALQRModel
from steergym.alqr.control import riccati_gain_rows, riccati_gains
from steergym.alqr.linearize import averaged_block_jacobians, block_jacobians
from steergym.alqr.models import _all_layer_last_token, _contrast_texts

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


SUBJECT = GeneratorRef("fake/alqr-subject")


# ================================================================== 1. Riccati recursion
def naive_riccati(A, q, r, q_f):
    # textbook (inv-based) reference: steer/lqr_utils.time_varying_lqr_noB
    L, n, _ = A.shape
    Q = torch.eye(n, dtype=A.dtype) * q
    R = torch.eye(n, dtype=A.dtype) * r
    S = torch.eye(n, dtype=A.dtype) * q_f
    K = torch.zeros(L, n, n, dtype=A.dtype)
    for t in reversed(range(L)):
        At = A[t]
        P = S + R
        F = S @ At
        G = Q + At.transpose(-2, -1) @ S @ At
        P_inv = torch.linalg.inv(P)
        K[t] = P_inv @ F
        S = G - F.transpose(-2, -1) @ P_inv @ F
    return K


g = torch.Generator().manual_seed(0)
A_rand = torch.randn(5, 6, 6, generator=g) * 0.3
q, r, q_f = 10.0, 10.0, 1.0

K_ref = naive_riccati(A_rand, q, r, q_f)
K_ours = riccati_gains(A_rand, q, r, q_f)
check("riccati_gains matches naive inv-based reference (1a)",
      torch.allclose(K_ours, K_ref, atol=1e-5))

V_rand = torch.randn(5, 6, generator=g)
W_ours = riccati_gain_rows(A_rand, V_rand, q, r, q_f)
check("riccati_gain_rows exactly equals riccati_gains[t] @ V[t] (1b, bitwise)",
      all(torch.equal(W_ours[t], K_ours[t] @ V_rand[t]) for t in range(5)))

# (1c) scalar case n=1, L=2, hand-derived closed form
a0, a1 = 0.7, -1.3
qs, rs, qfs = 2.0, 3.0, 0.5
A_scalar = torch.tensor([[[a0]], [[a1]]])
S2 = qfs
P1 = S2 + rs
F1 = S2 * a1
K1 = F1 / P1
S1 = (qs + a1 * a1 * S2) - (F1 * F1) / P1
P0 = S1 + rs
F0 = S1 * a0
K0 = F0 / P0
K_scalar = riccati_gains(A_scalar, qs, rs, qfs)
check("riccati_gains: scalar n=1,L=2 matches hand-derived closed form (1c)",
      abs(K_scalar[0, 0, 0].item() - K0) < 1e-6
      and abs(K_scalar[1, 0, 0].item() - K1) < 1e-6)

# (1d) A = 0 => K = 0
A_zero = torch.zeros(4, 5, 5)
K_zero = riccati_gains(A_zero, q, r, q_f)
check("riccati_gains: A=0 => K=0 (1d)", torch.allclose(K_zero, torch.zeros_like(K_zero)))

# (1e) r -> huge => ||K|| -> ~0
K_huge_r = riccati_gains(A_rand, q, 1e8, q_f)
check("riccati_gains: r -> huge => ||K|| -> ~0 (1e)", float(K_huge_r.norm()) < 1e-4)

# (1f) zero-norm guard (upstream's E_unit[i]=0 when nrm==0): e=0 => v=0 (no
# NaN from a 0/0 division), and w = riccati_gain_rows(A, v, ...) is then
# exactly 0 too -- the rank-1 identity, not a special case in the recursion
e_zero = torch.zeros(3, 5)
mu_zero = e_zero.norm(dim=-1)
zero_mask = mu_zero == 0
v_zero = torch.zeros_like(e_zero)
v_zero[~zero_mask] = e_zero[~zero_mask] / mu_zero[~zero_mask].unsqueeze(-1)
check("zero-norm guard: e=0 => mu=0, v=0, no NaN (1f)",
      bool(torch.all(mu_zero == 0)) and bool(torch.all(v_zero == 0))
      and not bool(torch.any(torch.isnan(v_zero))))
A_for_zero = torch.randn(3, 5, 5, generator=g)
w_zero = riccati_gain_rows(A_for_zero, v_zero, q, r, q_f)
check("zero-norm guard: v=0 => w=0 via riccati_gain_rows (1f)",
      bool(torch.all(w_zero == 0)))


# ================================================================== 2. ALQRContext closed-loop math
class FakeBlock(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, h):
        return (h + torch.tanh(h @ self.M),)


def run_stack(layers, h):
    for layer in layers:
        out = layer(h)
        h = out[0] if isinstance(out, tuple) else out
    return h


gctx = torch.Generator().manual_seed(1)
Lc, Dc, Bc, Tc = 3, 4, 2, 5
layers = [FakeBlock(torch.randn(Dc, Dc, generator=gctx) * 0.2) for _ in range(Lc)]
v_hand = torch.randn(Lc, Dc, generator=gctx)
v_hand = v_hand / v_hand.norm(dim=-1, keepdim=True)
w_hand = torch.randn(Lc, Dc, generator=gctx)
mu_hand = torch.rand(Lc, generator=gctx) + 0.5
lmbda = 1.7
h0 = torch.randn(Bc, Tc, Dc, generator=gctx)

baseline = run_stack(layers, h0.clone())


def hand_closed_loop(positions):
    x = h0.clone()
    norms = []
    for k in range(Lc):
        x_in = x
        out = layers[k](x_in)[0]
        beta_k = lmbda * mu_hand[k].item()
        if positions == "generated":
            xin_last = x_in[:, -1, :]
            alpha = beta_k - xin_last @ v_hand[k]     # [B]
            u = alpha[:, None] * w_hand[k]             # [B, d]
            out = out.clone()
            out[:, -1, :] = out[:, -1, :] + u
            norms.append(u.norm(dim=-1))
        else:
            alpha = beta_k - x_in @ v_hand[k]          # [B, T]
            u = alpha.unsqueeze(-1) * w_hand[k]         # [B, T, d]
            out = out + u
            norms.append(u.reshape(-1, Dc).norm(dim=-1))
        x = out
    return x, norms


hand_gen, norms_gen = hand_closed_loop("generated")
ctx = ALQRContext(layers, v_hand.clone(), w_hand.clone(), mu_hand.clone(), lmbda, positions="generated")
with ctx:
    steered_gen = run_stack(layers, h0.clone())
check("ALQRContext positions='generated': matches hand closed-loop recompute (2)",
      torch.allclose(steered_gen, hand_gen, atol=1e-5))
diag_gen = ctx.diagnostics()
all_norms_gen = torch.cat(norms_gen)
check("diagnostics steered_rows == layers x batch (2, generated)",
      diag_gen["steered_rows"] == Lc * Bc)
check("diagnostics mean/max match ||u|| by hand (2, generated)",
      abs(diag_gen["applied_delta_norm_mean"] - float(all_norms_gen.mean())) < 1e-5
      and abs(diag_gen["applied_delta_norm_max"] - float(all_norms_gen.max())) < 1e-5)
check("diagnostics() is idempotent when called again with no new forward (2, generated)",
      ctx.diagnostics() == diag_gen)

hand_all, norms_all = hand_closed_loop("all")
ctx_all = ALQRContext(layers, v_hand.clone(), w_hand.clone(), mu_hand.clone(), lmbda, positions="all")
with ctx_all:
    steered_all = run_stack(layers, h0.clone())
check("ALQRContext positions='all': matches hand closed-loop recompute (2)",
      torch.allclose(steered_all, hand_all, atol=1e-5))
diag_all = ctx_all.diagnostics()
check("diagnostics steered_rows == layers x batch x positions (2, all)",
      diag_all["steered_rows"] == Lc * Bc * Tc)
all_norms_all = torch.cat(norms_all)
check("diagnostics mean/max match ||u|| by hand (2, all)",
      abs(diag_all["applied_delta_norm_mean"] - float(all_norms_all.mean())) < 1e-5
      and abs(diag_all["applied_delta_norm_max"] - float(all_norms_all.max())) < 1e-5)
check("diagnostics() is idempotent when called again with no new forward (2, all)",
      ctx_all.diagnostics() == diag_all)

# diagnostics accumulate correctly across multiple forwards under one entry
# (not just one-shot): same input run twice should double steered_rows and
# leave the per-row mean/max unchanged.
ctx_multi = ALQRContext(layers, v_hand.clone(), w_hand.clone(), mu_hand.clone(),
                        lmbda, positions="generated")
with ctx_multi:
    run_stack(layers, h0.clone())
    diag_once = ctx_multi.diagnostics()
    run_stack(layers, h0.clone())
    diag_twice = ctx_multi.diagnostics()
check("diagnostics after a second forward: steered_rows doubles (2)",
      diag_twice["steered_rows"] == 2 * diag_once["steered_rows"])
check("diagnostics after a second forward: mean/max unchanged (identical repeated input) (2)",
      abs(diag_twice["applied_delta_norm_mean"] - diag_once["applied_delta_norm_mean"]) < 1e-6
      and abs(diag_twice["applied_delta_norm_max"] - diag_once["applied_delta_norm_max"]) < 1e-6)
check("diagnostics() called twice in a row with no intervening forward is idempotent (2)",
      ctx_multi.diagnostics() == diag_twice)

# _dev_rows per-layer device cache: populated once per layer, and the same
# (v_k, w_k) tensor objects are reused (identity, not just equal) across
# forwards on the same device. Cross-device re-caching (layers.device
# changing) isn't reachable on this CPU-only harness.
check("_dev_rows cache: one entry per layer after a forward (2)",
      set(ctx_multi._dev_rows) == set(range(Lc)))
cached_rows_before = {k: ctx_multi._dev_rows[k] for k in range(Lc)}
with ctx_multi:
    run_stack(layers, h0.clone())
cached_rows_after = {k: ctx_multi._dev_rows[k] for k in range(Lc)}
check("_dev_rows cache: same (v_k, w_k) tensor objects reused across forwards (2)",
      all(cached_rows_before[k][0] is cached_rows_after[k][0]
          and cached_rows_before[k][1] is cached_rows_after[k][1] for k in range(Lc)))

# float16 stream keeps dtype; _apply does a SINGLE rounding on the add
# ((h[..., -1, :].float() + u).to(h.dtype) / (h.float() + u).to(h.dtype)),
# not an approximate fp32 recompute cast back -- replicate that exact
# formula by hand (same op order) and compare bitwise.
layers_half = [FakeBlock(m.M.half()) for m in layers]
h0_half = h0.half()


def hand_closed_loop_exact(positions, layers_use, h0_use):
    # mirrors ALQRContext._apply's op order exactly (single rounding on the
    # add), so this is a bitwise formula check, not a tolerance-based one
    x = h0_use.clone()
    for k in range(Lc):
        x_in = x
        out = layers_use[k](x_in)[0]
        beta_k = lmbda * mu_hand[k].item()
        if positions == "generated":
            xin_last = x_in[:, -1, :].float()
            alpha = beta_k - xin_last @ v_hand[k]
            u = alpha[:, None] * w_hand[k]
            out = out.clone()
            out[:, -1, :] = (out[:, -1, :].float() + u).to(out.dtype)
        else:
            alpha = beta_k - x_in.float() @ v_hand[k]
            u = alpha.unsqueeze(-1) * w_hand[k]
            out = (out.float() + u).to(out.dtype)
        x = out
    return x


hand_half_gen = hand_closed_loop_exact("generated", layers_half, h0_half)
ctx_half = ALQRContext(layers_half, v_hand.clone(), w_hand.clone(), mu_hand.clone(),
                        lmbda, positions="generated")
with ctx_half:
    steered_half = run_stack(layers_half, h0_half.clone())
check("float16 stream: output stays float16", steered_half.dtype == torch.float16)
check("float16 stream: EXACT bitwise match to the single-rounding formula, positions='generated' (2)",
      torch.equal(steered_half, hand_half_gen))

hand_half_all = hand_closed_loop_exact("all", layers_half, h0_half)
ctx_half_all = ALQRContext(layers_half, v_hand.clone(), w_hand.clone(), mu_hand.clone(),
                           lmbda, positions="all")
with ctx_half_all:
    steered_half_all = run_stack(layers_half, h0_half.clone())
check("float16 stream: EXACT bitwise match to the single-rounding formula, positions='all' (2)",
      torch.equal(steered_half_all, hand_half_all))

# re-entry raises
ctx_re = ALQRContext(layers, v_hand.clone(), w_hand.clone(), mu_hand.clone(), lmbda)
ctx_re.__enter__()
reentry_raised = False
try:
    ctx_re.__enter__()
except RuntimeError:
    reentry_raised = True
ctx_re.__exit__(None, None, None)
check("ALQRContext: re-entry raises RuntimeError (2)", reentry_raised)

# after __exit__, forward is unmodified
after_exit = run_stack(layers, h0.clone())
check("ALQRContext: after __exit__ a forward is unmodified (2)",
      torch.allclose(after_exit, baseline))

try:
    ALQRContext(layers, v_hand, w_hand, mu_hand, 1.0, positions="bogus")
    bad_positions_raised = False
except ValueError:
    bad_positions_raised = True
check("ALQRContext: bad positions raises ValueError", bad_positions_raised)


# ================================================================== 3. Jacobians on a fake HF-like model
class FakeHFLayer(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, hidden, attention_mask=None, position_ids=None, position_embeddings=None):
        return hidden + torch.tanh(hidden @ self.M)


class FakeHFInner(nn.Module):
    def __init__(self, embed, layer_list):
        super().__init__()
        self.embed = embed
        self.layers = nn.ModuleList(layer_list)

    def rotary_emb(self, hidden, position_ids):
        return (torch.zeros_like(hidden), torch.zeros_like(hidden))  # ignored by FakeHFLayer


class FakeHFModel(nn.Module):
    def __init__(self, D, V, L, gen):
        super().__init__()
        embed = nn.Embedding(V, D)
        layer_list = [FakeHFLayer(torch.randn(D, D, generator=gen) * 0.15) for _ in range(L)]
        self.model = FakeHFInner(embed, layer_list)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        h = self.model.embed(input_ids)
        hs = [h]
        for layer in self.model.layers:
            h = layer(h)
            hs.append(h)
        return types.SimpleNamespace(hidden_states=tuple(hs))


g3 = torch.Generator().manual_seed(2)
D3, V3, L3, Seq3 = 6, 50, 3, 7
fake_hf = FakeHFModel(D3, V3, L3, g3)
for p in fake_hf.parameters():
    p.requires_grad_(False)
input_ids3 = torch.randint(0, V3, (1, Seq3), generator=g3)

A_got = block_jacobians(fake_hf, input_ids3)
check("block_jacobians: shape [L, d, d] (3)", tuple(A_got.shape) == (L3, D3, D3))

with torch.no_grad():
    hs3 = fake_hf(input_ids=input_ids3, output_hidden_states=True).hidden_states

all_analytic_ok = True
for k in range(L3):
    x_last = hs3[k][0, -1, :]
    M = fake_hf.model.layers[k].M
    y = x_last @ M
    s = 1.0 - torch.tanh(y) ** 2
    J_analytic = torch.eye(D3) + s.unsqueeze(1) * M.t()
    if not torch.allclose(A_got[k], J_analytic, atol=1e-5):
        all_analytic_ok = False
check("block_jacobians: matches analytic I + diag(1-tanh^2(xM)) @ M^T per layer (3)",
      all_analytic_ok)

# finite-difference spot check of a random column, one layer (central
# difference: O(eps^2) truncation, well clear of float32 noise at eps=1e-3)
k_fd, j_fd = 1, 2
eps = 1e-3
x_base = hs3[k_fd][0, -1, :].clone()
layer_fd = fake_hf.model.layers[k_fd]


def layer_last_out(x_last):
    return x_last + torch.tanh(x_last @ layer_fd.M)


x_plus, x_minus = x_base.clone(), x_base.clone()
x_plus[j_fd] += eps
x_minus[j_fd] -= eps
fd_col = (layer_last_out(x_plus) - layer_last_out(x_minus)) / (2 * eps)
check("block_jacobians: finite-difference spot check of a random column (3)",
      torch.allclose(A_got[k_fd][:, j_fd], fd_col, atol=2e-3))

# averaged_block_jacobians LEFT-truncates to the final max_ctx tokens: a
# text longer than max_ctx must give the SAME jacobian as calling
# block_jacobians directly on the tail ids, and a DIFFERENT jacobian than
# the head ids (ruling out a right-truncation regression).
g3b = torch.Generator().manual_seed(21)
MAX_CTX3 = 4
full_ids3 = torch.randint(0, V3, (2 * MAX_CTX3 + 3,), generator=g3b).tolist()
tail_ids3 = full_ids3[-MAX_CTX3:]
head_ids3 = full_ids3[:MAX_CTX3]


class _IdsTok:
    def __init__(self, id_map):
        self.id_map = id_map

    def __call__(self, text):
        return {"input_ids": self.id_map[text]}


ids_tok3 = _IdsTok({"long-text": full_ids3})
A_truncated = averaged_block_jacobians(fake_hf, ids_tok3, ["long-text"], MAX_CTX3)
A_tail_direct = block_jacobians(fake_hf, torch.tensor([tail_ids3], dtype=torch.long))
A_head_direct = block_jacobians(fake_hf, torch.tensor([head_ids3], dtype=torch.long))
check("averaged_block_jacobians: LEFT-truncation matches block_jacobians on the TAIL ids (3)",
      torch.allclose(A_truncated, A_tail_direct, atol=1e-6))
check("averaged_block_jacobians: LEFT-truncation does NOT match the HEAD ids (3)",
      not torch.allclose(A_truncated, A_head_direct, atol=1e-3))


# ================================================================== 4/5. Fit end-to-end + save/load, fake LM + char tokenizer
tf = types.ModuleType("transformers")
tf.set_seed = lambda s: torch.manual_seed(s)
sys.modules["transformers"] = tf

D4, V4, L4, BOS, EOS, PAD = 10, 2000, 3, 1, 2, 0


class FakeTok:
    padding_side = "right"
    bos_token = "<bos>"
    eos_token_id = EOS
    pad_token_id = PAD

    def _t(self, c):
        return 3 + ord(c)

    def decode(self, ids):
        return "".join(chr(int(i) - 3) for i in ids if int(i) > 2)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        s = "«" + "".join(m["content"] for m in messages) + "»"
        if not tokenize:
            return self.bos_token + s
        return [BOS] + [self._t(c) for c in s]

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        single = isinstance(texts, str)
        rows_in = [texts] if single else texts
        rows = []
        for t in rows_in:
            ids = [BOS] + [self._t(c) for c in t]
            if truncation and max_length:
                ids = ids[:max_length]
            rows.append(ids)
        if not padding and return_tensors is None:
            return {"input_ids": rows[0] if single else rows}
        T = max(len(r) for r in rows)
        ids_t = torch.full((len(rows), T), PAD, dtype=torch.long)
        mask = torch.zeros((len(rows), T), dtype=torch.long)
        for i, r in enumerate(rows):
            ids_t[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = 1
        return {"input_ids": ids_t, "attention_mask": mask}


class FakeLMLayer(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, hidden, attention_mask=None, position_ids=None, position_embeddings=None):
        return hidden + torch.tanh(hidden @ self.M)


class FakeLMInner(nn.Module):
    def __init__(self, D, V, L, gen):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.layers = nn.ModuleList(
            [FakeLMLayer(torch.randn(D, D, generator=gen) * 0.1) for _ in range(L)])

    def rotary_emb(self, hidden, position_ids):
        return (torch.zeros_like(hidden), torch.zeros_like(hidden))


class FakeLM(nn.Module):
    def __init__(self, D, V, L, gen):
        super().__init__()
        self.model = FakeLMInner(D, V, L, gen)
        self.config = types.SimpleNamespace(hidden_size=D)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        h = self.model.embed(input_ids)
        hs = [h]
        for layer in self.model.layers:
            h = layer(h)
            hs.append(h)
        return types.SimpleNamespace(hidden_states=tuple(hs))


g4 = torch.Generator().manual_seed(3)
lm = FakeLM(D4, V4, L4, g4)
for p in lm.parameters():
    p.requires_grad_(False)
fake_tok = FakeTok()
subject4 = GeneratorRef("fake/alqr-fit-subject")
gen4 = GeneratorLM(subject4, lm, fake_tok)

import polars as pl  # noqa: E402


def mk_row(concept_id, prompt, response, output_concept, category, genre="text"):
    return {
        "concept_id": concept_id,
        "prompt": prompt,
        "response": response,
        "meta": {"output_concept": output_concept, "category": category, "concept_genre": genre},
    }


rows4 = []
for i in range(6):
    rows4.append(mk_row(0, f"tell me about A{i}", "A" * (i + 3) + "!", "conceptA", "positive"))
for i in range(6):
    rows4.append(mk_row(1, f"tell me about B{i}", f"bland text response {i}", "conceptB", "positive"))
for i in range(8):
    rows4.append(mk_row(-1, f"neutral prompt {i}", f"a plain neutral reply {i}", "", "negative"))
samples4 = pl.DataFrame(rows4)
concepts4 = [Concept(id=0, name="conceptA", description=""), Concept(id=1, name="conceptB", description="")]

cfg4 = ALQRConfig(subject=subject4, batch_size=2, max_length=64, jacobian_prompts=4,
                  jacobian_max_ctx=32, seed=7)
model4 = ALQRModel(cfg4).bind(lm, fake_tok)
model4.fit(concepts4, samples4)

check("ALQRModel.fit: concept ids present", set(model4.concepts) == {0, 1})
for cid in (0, 1):
    c = model4.concepts[cid]
    check(f"ALQRModel.fit: concept {cid} v/w shape [{L4},{D4}] (4)",
          tuple(c["v"].shape) == (L4, D4) and tuple(c["w"].shape) == (L4, D4))
    check(f"ALQRModel.fit: concept {cid} mu shape [{L4}] (4)", tuple(c["mu"].shape) == (L4,))
    norms_v = c["v"].norm(dim=-1)
    check(f"ALQRModel.fit: concept {cid} every v row unit or zero (4)",
          bool(torch.all(((norms_v - 1.0).abs() < 1e-4) | (norms_v < 1e-6))))

# hand-compute concept 0's v/mu/w and compare
pos_texts0, neg_texts0 = _contrast_texts(samples4, concepts4[0], fake_tok)
pos_X0 = _all_layer_last_token(gen4, pos_texts0, cfg4)
neg_X0 = _all_layer_last_token(gen4, neg_texts0, cfg4)
e0 = pos_X0.mean(dim=0) - neg_X0.mean(dim=0)
mu0 = e0.norm(dim=-1)
zero0 = mu0 == 0
v0 = torch.zeros_like(e0)
v0[~zero0] = e0[~zero0] / mu0[~zero0].unsqueeze(-1)
jac_texts0 = random.Random(cfg4.seed).sample(pos_texts0, min(cfg4.jacobian_prompts, len(pos_texts0)))
A0 = averaged_block_jacobians(lm, fake_tok, jac_texts0, cfg4.jacobian_max_ctx)
w0 = riccati_gain_rows(A0, v0, cfg4.q, cfg4.r, cfg4.q_f)

check("ALQRModel.fit: concept 0 mu matches hand-computed mean-diff norms (4)",
      torch.allclose(model4.concepts[0]["mu"], mu0, atol=1e-5))
check("ALQRModel.fit: concept 0 v matches hand-computed unit directions (4)",
      torch.allclose(model4.concepts[0]["v"], v0, atol=1e-5))
check("ALQRModel.fit: concept 0 w matches riccati_gain_rows of the hand-averaged A (4)",
      torch.allclose(model4.concepts[0]["w"], w0, atol=1e-4))

# determinism across two fits (same seed)
model4b = ALQRModel(ALQRConfig(subject=subject4, batch_size=2, max_length=64, jacobian_prompts=4,
                               jacobian_max_ctx=32, seed=7)).bind(lm, fake_tok)
model4b.fit(concepts4, samples4)
check("ALQRModel.fit: deterministic across two fits, same seed (4)",
      torch.equal(model4.concepts[0]["v"], model4b.concepts[0]["v"])
      and torch.equal(model4.concepts[0]["w"], model4b.concepts[0]["w"])
      and torch.equal(model4.concepts[0]["mu"], model4b.concepts[0]["mu"]))

# context() on the real fitted model: generation-shaped forward differs only downstream of layer 0
ids0 = fake_tok(pos_texts0[:3], return_tensors="pt", padding=True)["input_ids"]
base_h4 = lm(input_ids=ids0).hidden_states[-1]
with model4.context(concepts4[0], lmbda=2.0, positions="generated") as ctx4:
    steered_h4 = lm(input_ids=ids0).hidden_states[-1]
check("ALQRModel.context: steers a real forward (final hidden state differs) (4)",
      not torch.allclose(steered_h4, base_h4))
diag4 = ctx4.diagnostics()
check("ALQRModel.context: diagnostics steered_rows == layers x batch (4)",
      diag4["steered_rows"] == L4 * 3)

# not-bound raises
unbound = ALQRModel(ALQRConfig(subject=subject4))
try:
    unbound.context(concepts4[0])
    not_bound_raised = False
except RuntimeError:
    not_bound_raised = True
check("ALQRModel.context: not bound raises RuntimeError (4)", not_bound_raised)

# fit() raises ValueError when a concept has no positive or no negative rows
# (separate ALQRModel instances -- fit() resets self.concepts, and we don't
# want to disturb model4's already-fitted state used above/below)
concept_no_match = Concept(id=5, name="no-such-concept", description="")
model_empty_pos = ALQRModel(cfg4).bind(lm, fake_tok)
try:
    model_empty_pos.fit([concept_no_match], samples4)
    empty_pos_raised = False
except ValueError:
    empty_pos_raised = True
check("ALQRModel.fit: concept with zero positive rows raises ValueError (4)", empty_pos_raised)

rows4_no_neg = [r for r in rows4 if r["meta"]["category"] != "negative"]
samples4_no_neg = pl.DataFrame(rows4_no_neg)
model_empty_neg = ALQRModel(cfg4).bind(lm, fake_tok)
try:
    model_empty_neg.fit(concepts4, samples4_no_neg)
    empty_neg_raised = False
except ValueError:
    empty_neg_raised = True
check("ALQRModel.fit: concept with zero negative rows raises ValueError (4)", empty_neg_raised)

# context() raises ValueError when the artifact's layer count mismatches
# the bound model's decoder-layer count
model_mismatch = ALQRModel(cfg4).bind(lm, fake_tok)
model_mismatch.concepts[0] = {
    "v": torch.zeros(L4 + 1, D4), "w": torch.zeros(L4 + 1, D4), "mu": torch.zeros(L4 + 1),
}
try:
    model_mismatch.context(concepts4[0])
    layer_mismatch_raised = False
except ValueError:
    layer_mismatch_raised = True
check("ALQRModel.context: artifact layer count != bound model raises ValueError (4)",
      layer_mismatch_raised)

# ---------------------------------------------------------------- 5. save/load roundtrip
import tempfile

SP = os.environ.get("ALQR_SCRATCH") or tempfile.mkdtemp(prefix="test_alqr_")
ckpt_path = f"{SP}/test_alqr_ckpt.pt"
model4.save(ckpt_path)
loaded4 = ALQRModel.load(ckpt_path)
check("save/load roundtrip: concept ids preserved (5)", set(loaded4.concepts) == set(model4.concepts))
check("save/load roundtrip: v/w/mu tensors equal (5)", all(
    torch.equal(loaded4.concepts[cid]["v"], model4.concepts[cid]["v"])
    and torch.equal(loaded4.concepts[cid]["w"], model4.concepts[cid]["w"])
    and torch.equal(loaded4.concepts[cid]["mu"], model4.concepts[cid]["mu"])
    for cid in model4.concepts))
check("save/load roundtrip: config fields preserved (5)",
      loaded4.config.q == cfg4.q and loaded4.config.r == cfg4.r and loaded4.config.q_f == cfg4.q_f
      and loaded4.config.seed == cfg4.seed and loaded4.config.subject == cfg4.subject)

# wrong-cls artifact raises
bad_cls_path = f"{SP}/test_alqr_badcls.pt"
torch.save({"cls": "NotALQRModel", "config": {}, "concepts": {}}, bad_cls_path)
try:
    ALQRModel.load(bad_cls_path)
    wrong_cls_raised = False
except ValueError:
    wrong_cls_raised = True
check("save/load: wrong-cls artifact raises ValueError (5)", wrong_cls_raised)

# buffer-key mismatch raises
bad_keys_path = f"{SP}/test_alqr_badkeys.pt"
cfg_d = {**{"q": 1.0, "r": 1.0, "q_f": 1.0, "jacobian_prompts": 1, "jacobian_max_ctx": 8,
           "batch_size": 1, "max_length": 8, "seed": 1, "layer": None, "site": None},
         "subject": {"model_id": "x", "revision": None}}
torch.save({
    "cls": "ALQRModel", "config": cfg_d,
    "concepts": {0: {"v": torch.zeros(2, 3), "w": torch.zeros(2, 3)}},  # missing "mu"
}, bad_keys_path)
try:
    ALQRModel.load(bad_keys_path)
    bad_keys_raised = False
except ValueError:
    bad_keys_raised = True
check("save/load: missing-buffer-key artifact raises ValueError (5)", bad_keys_raised)


# ================================================================== 6. ALQR_DIR-gated source parity
ALQR_DIR = os.environ.get("ALQR_DIR")
if not ALQR_DIR:
    print("SKIP upstream source-parity leg: ALQR_DIR not set")
else:
    sys.path.insert(0, ALQR_DIR)
    try:
        del sys.modules["transformers"]  # drop our stub; the real repo needs the real package
    except KeyError:
        pass
    try:
        import steer.lqr_utils as upstream_lqr
    except Exception as exc:  # matplotlib / sklearn / missing deps etc.
        print(f"SKIP upstream source-parity leg: import failed ({exc!r})")
        upstream_lqr = None

    if upstream_lqr is not None:
        gp = torch.Generator().manual_seed(4)
        Ap = torch.randn(3, 8, 8, generator=gp) * 0.3
        qp, rp, qfp = 0.7, 2.3, 0.4
        Qp = torch.eye(8).unsqueeze(0).repeat(3, 1, 1) * qp
        Rp = torch.eye(8).unsqueeze(0).repeat(3, 1, 1) * rp
        Sfp = torch.eye(8) * qfp
        K_upstream = upstream_lqr.time_varying_lqr_noB_mem_efficient(Ap, Qp, Rp, Sfp)
        K_ours_p = riccati_gains(Ap, qp, rp, qfp)
        check("upstream parity: riccati_gains vs time_varying_lqr_noB_mem_efficient (6)",
              torch.allclose(K_ours_p, K_upstream, atol=1e-6))

        try:
            import steer.steering as upstream_steering

            lqr_obj = object.__new__(upstream_steering.LQRSteering)
            Lp, n = 3, 8
            lqr_obj.K = K_upstream
            lqr_obj.T = Lp
            lqr_obj.n = n
            lqr_obj.device = torch.device("cpu")
            lqr_obj.mode = None
            lqr_obj.setpoint_type = "linear"
            lqr_obj.ALL_TOKENS = False
            E = torch.randn(Lp, n, generator=gp)
            lqr_obj.E = E
            lqr_obj.E_unit = torch.zeros_like(E)
            lqr_obj.betas = [0.0] * Lp
            lmbda_p = 1.3
            for i, e in enumerate(E):
                nrm = torch.linalg.norm(e)
                if nrm == 0:
                    continue
                lqr_obj.E_unit[i] = e / nrm
                lqr_obj.betas[i] = lmbda_p * float(nrm)
            lqr_obj.X = torch.zeros(Lp + 1, n)
            lqr_obj.U = torch.zeros(Lp, n)

            layer_idx = 0
            Bp = 2
            x_in = torch.randn(1, Bp, n, generator=gp)
            fake_output = (torch.randn(1, Bp, n, generator=gp).clone(),)
            upstream_out = lqr_obj.hook_setpoint_tracking(layer_idx, None, (x_in,), tuple(t.clone() for t in fake_output))
            upstream_delta = (upstream_out[0] - fake_output[0])[..., -1, :]

            v_p = lqr_obj.E_unit[layer_idx]
            w_p = (K_upstream[layer_idx] @ v_p)
            beta_p = lqr_obj.betas[layer_idx]
            x_last = x_in[:, -1, :]
            alpha_p = beta_p - x_last @ v_p
            ours_delta = alpha_p[:, None] * w_p

            check("upstream parity: our hook delta matches hook_setpoint_tracking (6, best-effort)",
                  torch.allclose(ours_delta, upstream_delta.squeeze(0), atol=1e-5))

            # ALSO run our real ALQRContext (not just the hand formula above)
            # on a fake single-layer module stack with the SAME v/w/mu,
            # feeding it the identical fixed pre-steer output upstream's
            # hook was given, and compare its applied delta against
            # upstream's hook output. Both sides are plain fp32 here (no
            # half-precision stream), so _apply's single-rounding add is
            # exact and this is a bitwise comparison, not a tolerance one.
            mu_for_ctx = torch.linalg.norm(E[layer_idx]).unsqueeze(0)       # [1]
            v_for_ctx = v_p.unsqueeze(0).clone()                            # [1, n]
            w_for_ctx = w_p.unsqueeze(0).clone()                            # [1, n]

            class _FixedOutputBlock(nn.Module):
                def __init__(self, fixed_out):
                    super().__init__()
                    self.fixed_out = fixed_out

                def forward(self, h):
                    return (self.fixed_out,)

            fixed_layer = _FixedOutputBlock(fake_output[0].clone())
            our_ctx_p = ALQRContext([fixed_layer], v_for_ctx, w_for_ctx, mu_for_ctx,
                                    lmbda_p, positions="generated")
            with our_ctx_p:
                our_out_p = fixed_layer(x_in.clone())
            our_delta_p = (our_out_p[0] - fake_output[0])[..., -1, :]

            # NOT bitwise-identical to upstream in general: upstream applies
            # K_t @ (alpha * v) as a matmul while we apply alpha * (K_t @ v)
            # (the rank-1 identity, precomputed as w) -- mathematically
            # equal, but floating point is non-associative, so this is the
            # same atol as the hand-formula check above, not torch.equal.
            # Both sides are plain fp32 (no half-precision stream here), so
            # _apply's single-vs-double rounding distinction is a no-op
            # ((h.float() + u).to(h.dtype) when h.dtype is already float32)
            # and the only source of disagreement is that associativity.
            check("upstream parity: our real ALQRContext delta matches hook_setpoint_tracking "
                  "(6, best-effort, fp32)",
                  torch.allclose(our_delta_p.squeeze(0), upstream_delta.squeeze(0), atol=1e-5))
        except Exception as exc:
            print(f"SKIP upstream hook_setpoint_tracking comparison: {exc!r}")


# ================================================================== 7. real-HF-layer leg (tiny random Llama/Qwen2, offline)
if _real_transformers is None:
    print("SKIP real-HF leg: transformers unavailable")
else:
    HF_SEED = 123
    HF_VOCAB = 300
    HF_D = 32
    HF_LAYERS = 2
    HF_HEADS = 4
    HF_KV_HEADS = 2

    def _mk_real_model(cfg_cls, model_cls, **extra):
        torch.manual_seed(HF_SEED)
        cfg = cfg_cls(vocab_size=HF_VOCAB, hidden_size=HF_D, intermediate_size=2 * HF_D,
                      num_hidden_layers=HF_LAYERS, num_attention_heads=HF_HEADS,
                      num_key_value_heads=HF_KV_HEADS, max_position_embeddings=64,
                      pad_token_id=PAD, attn_implementation="eager", **extra)
        m = model_cls(cfg).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    llama7 = _mk_real_model(_RealLlamaConfig, _RealLlamaForCausalLM)
    qwen7 = _mk_real_model(_RealQwen2Config, _RealQwen2ForCausalLM)

    # ---- 7a. closed-loop propagation: layer k+1's real forward INPUT must
    # equal layer k's STEERED output. Capture hooks are registered AFTER
    # ALQRContext's (i.e. later in each module's _forward_hooks), so they
    # observe the already-steered `output` -- this exercises the real hook
    # wiring (does the model loop actually consume our hook's return
    # value?), not a hand replay. Both positions modes.
    def _closed_loop_check(name, model):
        layers7 = model.model.layers
        Lh = len(layers7)
        Dh = model.config.hidden_size
        gh = torch.Generator().manual_seed(31)
        v7 = torch.randn(Lh, Dh, generator=gh)
        v7 = v7 / v7.norm(dim=-1, keepdim=True)
        w7 = torch.randn(Lh, Dh, generator=gh) * 0.5
        mu7 = torch.rand(Lh, generator=gh) + 1.0
        lmbda7 = 1.3
        ids7 = torch.randint(3, HF_VOCAB, (1, 9), generator=gh)
        for positions in ("generated", "all"):
            seen = {}

            def _mk_capture(k):
                # a forward hook must return None (or leave it implicit) to
                # avoid REPLACING the real output -- a lambda ending in
                # dict.setdefault(...) returns the stored value, which
                # would silently corrupt every layer's forward output
                def _capture(mod, a, o):
                    h = o[0] if isinstance(o, tuple) else o
                    seen[k] = (a[0].detach().clone(), h.detach().clone())
                return _capture

            # ctx7 must __enter__ (register its hooks) BEFORE the capture
            # hooks below are registered, so the capture hooks run SECOND
            # in each module's hook chain and observe the already-steered
            # output (torch replaces `output` for later hooks whenever an
            # earlier one returns non-None)
            ctx7 = ALQRContext(layers7, v7.clone(), w7.clone(), mu7.clone(), lmbda7,
                               positions=positions)
            ctx7.__enter__()
            caps = [lay.register_forward_hook(_mk_capture(k))
                    for k, lay in enumerate(layers7)]
            with torch.no_grad():
                model(input_ids=ids7, attention_mask=torch.ones_like(ids7))
            for c in caps:
                c.remove()
            ctx7.__exit__(None, None, None)
            ok = all(torch.equal(seen[k][1], seen[k + 1][0]) for k in range(Lh - 1))
            check(f"{name}: closed-loop propagation (layer k+1 input == steered layer k "
                  f"output), positions={positions!r} (7)", ok)

    _closed_loop_check("real-HF llama", llama7)
    _closed_loop_check("real-HF qwen2", qwen7)

    # ---- 7b. KV-cache generate equivalence: a prompt's generated ids must
    # be identical whether it runs alone or left-padded inside a batch with
    # a longer prompt (min_new_tokens pins the length so a random model
    # emitting an "eos" id by chance can't shorten one side only).
    def _kvcache_check(name, model):
        layers7 = model.model.layers
        Lh = len(layers7)
        Dh = model.config.hidden_size
        gh = torch.Generator().manual_seed(32)
        v7 = torch.randn(Lh, Dh, generator=gh)
        v7 = v7 / v7.norm(dim=-1, keepdim=True)
        w7 = torch.randn(Lh, Dh, generator=gh) * 0.3
        mu7 = torch.rand(Lh, generator=gh) + 0.5
        lmbda7 = 1.1
        long_ids = torch.randint(3, HF_VOCAB, (1, 7), generator=gh)
        short_ids = torch.randint(3, HF_VOCAB, (1, 4), generator=gh)
        pad7 = torch.zeros(1, 3, dtype=torch.long)
        batch7 = torch.cat([torch.cat([pad7, short_ids], 1), long_ids], 0)
        am7 = torch.cat([torch.cat([torch.zeros(1, 3, dtype=torch.long),
                                    torch.ones(1, 4, dtype=torch.long)], 1),
                        torch.ones(1, 7, dtype=torch.long)], 0)

        def _gen(ids, am):
            ctx7 = ALQRContext(layers7, v7.clone(), w7.clone(), mu7.clone(), lmbda7,
                               positions="generated")
            with ctx7, torch.no_grad():
                return model.generate(input_ids=ids, attention_mask=am, max_new_tokens=4,
                                      min_new_tokens=4, do_sample=False, pad_token_id=0)

        out_batch = _gen(batch7, am7)
        out_alone = _gen(short_ids, torch.ones_like(short_ids))
        check(f"{name}: KV-cache generate, left-padded batch row0 == same prompt alone (7)",
              torch.equal(out_batch[0, 7:], out_alone[0, 4:]))

    _kvcache_check("real-HF llama", llama7)
    _kvcache_check("real-HF qwen2", qwen7)

    # ---- 7c. block_jacobians vs the model's own forward: the linearization
    # point (hidden_states[k]) must match a hook-captured layer-k INPUT, and
    # a finite-difference column of a real layer's last-position Jacobian
    # must match block_jacobians' analytic column.
    def _jacobian_check(name, model):
        layers7 = model.model.layers
        Lh = len(layers7)
        gh = torch.Generator().manual_seed(33)
        ids7 = torch.randint(3, HF_VOCAB, (1, 9), generator=gh)
        with torch.no_grad():
            hs7 = model(input_ids=ids7, attention_mask=torch.ones_like(ids7),
                       output_hidden_states=True).hidden_states

        captured = {}
        caps = [lay.register_forward_hook(
                    lambda mod, a, o, k=k: captured.update({k: a[0].detach().clone()}))
                for k, lay in enumerate(layers7)]
        with torch.no_grad():
            model(input_ids=ids7, attention_mask=torch.ones_like(ids7))
        for c in caps:
            c.remove()
        inputs_match = all(torch.equal(captured[k], hs7[k]) for k in range(Lh))
        check(f"{name}: hidden_states[k] matches hook-captured layer-k input (7)", inputs_match)

        A7 = block_jacobians(model, ids7)
        S7 = ids7.shape[1]
        pos7 = torch.arange(S7).unsqueeze(0)
        mask7 = torch.full((1, 1, S7, S7), torch.finfo(hs7[0].dtype).min).triu(1)
        k_fd7, j_fd7 = min(1, Lh - 1), 3

        def _lay_last(x):
            pe = model.model.rotary_emb(x, pos7)
            o = layers7[k_fd7](x, attention_mask=mask7, position_ids=pos7,
                               position_embeddings=pe)
            o = o[0] if isinstance(o, tuple) else o
            return o[0, -1]

        eps7 = 1e-3
        xp7 = hs7[k_fd7].clone()
        xp7[0, -1, j_fd7] += eps7
        xm7 = hs7[k_fd7].clone()
        xm7[0, -1, j_fd7] -= eps7
        with torch.no_grad():
            fd7 = (_lay_last(xp7) - _lay_last(xm7)) / (2 * eps7)
        check(f"{name}: block_jacobians finite-difference column check, layer {k_fd7} (7)",
              torch.allclose(fd7, A7[k_fd7][:, j_fd7], atol=5e-3))

    _jacobian_check("real-HF llama", llama7)
    _jacobian_check("real-HF qwen2", qwen7)

    # ---- 7d. _all_layer_last_token de-tautologized: batched right-padded
    # extraction compared against per-text UNPADDED single forwards (not a
    # same-shape self-consistency check, which can't catch e.g. an
    # off-by-one length index or a dead attention mask)
    class _SimpleIdsTok:
        padding_side = "right"
        pad_token_id = PAD
        eos_token_id = EOS

        def __call__(self, text, max_length=None, truncation=None):
            ids = [BOS] + [((ord(c) % 250) + 4) for c in text]
            if truncation and max_length:
                ids = ids[:max_length]
            return {"input_ids": ids}

    ids_tok7 = _SimpleIdsTok()
    gen7_ns = types.SimpleNamespace(tokenizer=ids_tok7, model=llama7)
    cfg7_detaut = ALQRConfig(subject=SUBJECT, batch_size=3, max_length=64)
    texts7 = ["a", "bb bb", "ccc ccc ccc"]
    batched7 = _all_layer_last_token(gen7_ns, texts7, cfg7_detaut)
    layers_llama7 = llama7.model.layers
    detaut_ok = True
    for i, t in enumerate(texts7):
        ids_i = torch.tensor([ids_tok7(t)["input_ids"]], dtype=torch.long)
        with torch.no_grad():
            hs_i = llama7(input_ids=ids_i, attention_mask=torch.ones_like(ids_i),
                         output_hidden_states=True).hidden_states
        expected_i = torch.stack([hs_i[k][0, -1, :] for k in range(len(layers_llama7))], dim=0)
        detaut_ok &= bool(torch.allclose(batched7[i], expected_i, atol=1e-5))
    check("_all_layer_last_token: batched right-padded rows match per-text UNPADDED "
          "single forwards (7)", detaut_ok)

    # ---- 7e. end-to-end fit on the tiny real llama: 2 concepts, tiny
    # jacobian budget, shapes + determinism + a context() generation under
    # torch.no_grad (this REPLACES relying on the fake-LM fit check alone
    # for wiring -- that check at section 4/5 stays in place too)
    rows7 = []
    for i in range(4):
        rows7.append(mk_row(0, f"tell me about X{i}", "X" * (i + 2) + "!", "conceptX", "positive"))
    for i in range(4):
        rows7.append(mk_row(1, f"tell me about Y{i}", f"plain text {i}", "conceptY", "positive"))
    for i in range(4):
        rows7.append(mk_row(-1, f"neutral prompt {i}", f"a neutral reply {i}", "", "negative"))
    samples7 = pl.DataFrame(rows7)
    concepts7 = [Concept(id=0, name="conceptX", description=""),
                Concept(id=1, name="conceptY", description="")]

    cfg7 = ALQRConfig(subject=SUBJECT, batch_size=2, max_length=32,
                      jacobian_prompts=2, jacobian_max_ctx=16, seed=5)
    model7 = ALQRModel(cfg7).bind(llama7, fake_tok)
    model7.fit(concepts7, samples7)

    Lh7 = len(llama7.model.layers)
    Dh7 = llama7.config.hidden_size
    check("real-HF fit: concept ids present (7)", set(model7.concepts) == {0, 1})
    for cid in (0, 1):
        c7 = model7.concepts[cid]
        check(f"real-HF fit: concept {cid} v/w shape [{Lh7},{Dh7}] (7)",
              tuple(c7["v"].shape) == (Lh7, Dh7) and tuple(c7["w"].shape) == (Lh7, Dh7))
        check(f"real-HF fit: concept {cid} mu shape [{Lh7}] (7)", tuple(c7["mu"].shape) == (Lh7,))

    model7b = ALQRModel(ALQRConfig(subject=SUBJECT, batch_size=2, max_length=32,
                                  jacobian_prompts=2, jacobian_max_ctx=16, seed=5)
                        ).bind(llama7, fake_tok)
    model7b.fit(concepts7, samples7)
    check("real-HF fit: deterministic across two fits, same seed (7)",
          torch.equal(model7.concepts[0]["v"], model7b.concepts[0]["v"])
          and torch.equal(model7.concepts[0]["w"], model7b.concepts[0]["w"])
          and torch.equal(model7.concepts[0]["mu"], model7b.concepts[0]["mu"]))

    gen_ok = True
    try:
        with model7.context(concepts7[0], lmbda=1.5, positions="generated") as ctx7e, \
                torch.no_grad():
            ids7e = torch.randint(3, HF_VOCAB, (1, 5))
            llama7(input_ids=ids7e, attention_mask=torch.ones_like(ids7e))
    except Exception:
        gen_ok = False
    check("real-HF fit: context() generation runs under torch.no_grad without error (7)", gen_ok)


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
