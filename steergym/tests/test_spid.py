#!/usr/bin/env python3
# Offline tests for steergym.alqr.spid (S-PID): the SPIDContext closed-loop
# controller math + its upstream quirks (accumulate-then-reset integral,
# cross-decode-step D-term leak, per-call state reset seam), SPIDModel
# fit/from_alqr/save/load. CPU torch is enough. Run:
#   PYTHONPATH=<pylibs>:<torchlibs> python3 steergym/tests/test_spid.py
# An optional ALQR_DIR-gated leg replays upstream's own hook_setpoint_tracking
# from the vendored lqrup source repo.
import os
import sys
import types
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

# See test_alqr.py for why this must happen before section D installs a
# fake-LM-only `transformers` stub.
try:
    import transformers as _real_transformers
    _RealLlamaConfig = _real_transformers.LlamaConfig
    _RealLlamaForCausalLM = _real_transformers.LlamaForCausalLM
except ImportError:
    _real_transformers = None

from steergym.concept import Concept
from steergym.generator import GeneratorLM, GeneratorRef
from steergym.alqr import ALQRConfig, ALQRModel, SPIDConfig, SPIDContext, SPIDModel
from steergym.alqr.models import _all_layer_last_token, _contrast_texts

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


SUBJECT = GeneratorRef("fake/spid-subject")


class FakeBlock(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, h):
        return (h + torch.tanh(h @ self.M),)


def run_one_layer(layer, x_in):
    out = layer(x_in)
    return out[0] if isinstance(out, tuple) else out


# ============================================================ A. closed-loop + quirks (a)/(b)
gA = torch.Generator().manual_seed(100)
LA, DA, BA, TA = 12, 4, 2, 5
layersA = [FakeBlock(torch.randn(DA, DA, generator=gA) * 0.2) for _ in range(LA)]
vA = torch.randn(LA, DA, generator=gA)
vA = vA / vA.norm(dim=-1, keepdim=True)
muA = torch.rand(LA, generator=gA) + 0.5
kpA, kiA, kdA = 2.0, 3.0, 0.5
lmbdaA = 1.4
h0A = torch.randn(BA, TA, DA, generator=gA)

ctxA = SPIDContext(layersA, vA.clone(), muA.clone(), kpA, kiA, kdA, lmbdaA,
                   integral_reset_every=10)


def hand_compute(layers, v, mu, kp, ki, kd, lmbda, h0, reset_every=10):
    x = h0.clone()
    e_sum = torch.zeros(h0.shape[0], h0.shape[-1])
    e_prev = torch.zeros_like(e_sum)
    outs, states = {}, {}
    for k, layer in enumerate(layers):
        x_in = x.clone()
        out = run_one_layer(layer, x_in)
        beta_k = lmbda * mu[k].item()
        alpha = beta_k - x_in[:, -1, :] @ v[k]
        e = alpha[:, None] * v[k]
        e_sum = e_sum + e
        if reset_every and k % reset_every == 0:
            e_sum = e_sum * 0
        u = kp * e + ki * e_sum + kd * (e - e_prev)
        e_prev = e
        out = out.clone()
        out[:, -1, :] = out[:, -1, :] + u
        x = out
        outs[k] = out.clone()
        states[k] = (e_sum.clone(), e_prev.clone())
    return outs, states


outs_hand_A, states_hand_A = hand_compute(layersA, vA, muA, kpA, kiA, kdA, lmbdaA, h0A)

captured_in_A, captured_state_after_A = {}, {}
with ctxA:
    x = h0A.clone()
    for k, layer in enumerate(layersA):
        x_in = x.clone()
        x = run_one_layer(layer, x_in)
        captured_in_A[k] = x_in
        captured_state_after_A[k] = (ctxA._state[0].clone(), ctxA._state[1].clone())

final_match = all(torch.allclose((captured_in_A[k + 1] if k + 1 < LA else x),
                                  outs_hand_A[k], atol=1e-5) for k in range(LA))
check("A: SPIDContext matches hand-recomputed closed loop at every layer (exact)", final_match)
check("A: SPIDContext internal (e_sum, e_prev) state matches hand recompute at every layer",
      all(torch.allclose(captured_state_after_A[k][0], states_hand_A[k][0], atol=1e-5)
          and torch.allclose(captured_state_after_A[k][1], states_hand_A[k][1], atol=1e-5)
          for k in range(LA)))

# quirk (a): layer reset_every+1 (=11) sees e_sum == its own e only (reset at layer 10
# wiped everything accumulated before it, including layer 10's own e)
beta11_A = lmbdaA * muA[11].item()
x11_A = captured_in_A[11][:, -1, :]
alpha11_A = beta11_A - x11_A @ vA[11]
e11_A = alpha11_A[:, None] * vA[11]
check("A quirk (a): layer reset_every+1 sees e_sum == its own e only",
      torch.allclose(captured_state_after_A[11][0], e11_A, atol=1e-6))

beta10_A = lmbdaA * muA[10].item()
x10_A = captured_in_A[10][:, -1, :]
alpha10_A = beta10_A - x10_A @ vA[10]
e10_A = alpha10_A[:, None] * vA[10]
check("A quirk (a) counter-evidence: e_sum at layer 11 excludes layer 10's e (the WRONG "
      "pre-accumulate-reset variant would include it)",
      not torch.allclose(captured_state_after_A[11][0], e10_A + e11_A, atol=1e-4))

# quirk (b): layer 0 (a reset layer) has exactly-zero e_sum after its own hook fires,
# including on a SECOND pass through the same context where state carried in nonzero
# (layer 11 is not a reset layer -- state after it is (e_sum=e11, e_prev=e11), both nonzero)
check("A quirk (b): layer 0 e_sum is exactly zero after its hook (reset layer), pass 1",
      bool(torch.all(captured_state_after_A[0][0] == 0)))
check("A: pre-condition for the carry-then-wipe check -- pass 1 ends with NONZERO carried state",
      bool(torch.any(captured_state_after_A[11][0] != 0))
      and bool(torch.any(captured_state_after_A[11][1] != 0)))

h1A = torch.randn(BA, 1, DA, generator=gA)  # a "decode step": T=1, NOT a new prefill
captured_state_after_A2 = {}
with ctxA:
    x2 = h1A.clone()
    for k, layer in enumerate(layersA):
        x_in2 = x2.clone()
        x2 = run_one_layer(layer, x_in2)
        captured_state_after_A2[k] = (ctxA._state[0].clone(), ctxA._state[1].clone())
check("A quirk (b): layer 0 e_sum wiped to zero on pass 2 despite nonzero carried-in state",
      bool(torch.all(captured_state_after_A2[0][0] == 0)))


# ============================================================ B. kp/ki/kd isolation + reset_every disable
DB = 4
vB = torch.randn(1, DB)
vB = vB / vB.norm(dim=-1, keepdim=True)
muB = torch.rand(1) + 0.4
x_in_B = torch.randn(3, 1, DB)
h_in_B = torch.randn(3, 1, DB)


def mk_single_layer_ctx(kp, ki, kd, e_sum_init, e_prev_init, reset_every=0):
    ctx = SPIDContext([None], vB.clone(), muB.clone(), kp, ki, kd, 1.0,
                      integral_reset_every=reset_every)
    ctx._state = (e_sum_init.clone(), e_prev_init.clone())
    ctx._batch_size = 3
    return ctx


e_sum_fixed = torch.randn(3, DB)
e_prev_1 = torch.randn(3, DB)
e_prev_2 = torch.randn(3, DB) * 5 + 2  # deliberately very different from e_prev_1

ctx_kd0_a = mk_single_layer_ctx(1.3, 0.7, 0.0, e_sum_fixed, e_prev_1)
ctx_kd0_b = mk_single_layer_ctx(1.3, 0.7, 0.0, e_sum_fixed, e_prev_2)
out_kd0_a = ctx_kd0_a._apply(0, x_in_B, h_in_B.clone())
out_kd0_b = ctx_kd0_b._apply(0, x_in_B, h_in_B.clone())
check("B: kd=0 kills the D term exactly -- e_prev value has NO effect on the output",
      torch.equal(out_kd0_a, out_kd0_b))

e_sum_1 = torch.randn(3, DB)
e_sum_2 = torch.randn(3, DB) * 5 + 2  # deliberately very different from e_sum_1
e_prev_fixed = torch.randn(3, DB)
ctx_ki0_a = mk_single_layer_ctx(1.3, 0.0, 0.9, e_sum_1, e_prev_fixed)
ctx_ki0_b = mk_single_layer_ctx(1.3, 0.0, 0.9, e_sum_2, e_prev_fixed)
out_ki0_a = ctx_ki0_a._apply(0, x_in_B, h_in_B.clone())
out_ki0_b = ctx_ki0_b._apply(0, x_in_B, h_in_B.clone())
check("B: ki=0 kills the integral term exactly -- e_sum value has NO effect on the output",
      torch.equal(out_ki0_a, out_ki0_b))

# positive-case arithmetic: kp=1,ki=0,kd=0 (pure P) => applied delta == e exactly
ctx_pureP = mk_single_layer_ctx(1.0, 0.0, 0.0, e_sum_1, e_prev_fixed)
beta_B = 1.0 * muB[0].item()
x_last_B = x_in_B[:, -1, :]
alpha_B = beta_B - x_last_B @ vB[0]
e_B = alpha_B[:, None] * vB[0]
out_pureP = ctx_pureP._apply(0, x_in_B, h_in_B.clone())
check("B: pure-P (kp=1, ki=kd=0) applies exactly e",
      torch.allclose((out_pureP[:, -1, :] - h_in_B[:, -1, :]).float(), e_B, atol=1e-6))

# integral_reset_every=0 disables the mod-10 wipe even at layer 0 (used above implicitly --
# ctx_kd0_a/b used reset_every=0 and layer 0 with a NONZERO e_sum_init that survived into
# the integral term; demonstrate explicitly against a reset_every=10 twin)
ctx_noreset = mk_single_layer_ctx(0.0, 1.0, 0.0, e_sum_fixed, e_prev_1, reset_every=0)
ctx_withreset = mk_single_layer_ctx(0.0, 1.0, 0.0, e_sum_fixed, e_prev_1, reset_every=10)
out_noreset = ctx_noreset._apply(0, x_in_B, h_in_B.clone())
out_withreset = ctx_withreset._apply(0, x_in_B, h_in_B.clone())
check("B: integral_reset_every=0 disables the layer-0 wipe (differs from reset_every=10)",
      not torch.allclose(out_noreset, out_withreset, atol=1e-6))
check("B: integral_reset_every=0 -- pure-I output includes the carried-in e_sum_fixed",
      torch.allclose((out_noreset[:, -1, :] - h_in_B[:, -1, :]).float(), e_B + e_sum_fixed,
                     atol=1e-5))
check("B: integral_reset_every=10 (layer 0) -- pure-I output is exactly zero "
      "(e_sum_fixed AND this layer's own e are both wiped by the layer-0 reset)",
      torch.allclose((out_withreset[:, -1, :] - h_in_B[:, -1, :]).float(),
                     torch.zeros_like(e_B), atol=1e-5))


# ============================================================ C. context lifecycle
try:
    SPIDContext([None], vB, muB, 1.0, 1.0, 1.0, 1.0, positions="all")
    pos_all_raised = False
except ValueError:
    pos_all_raised = True
check("C: positions='all' raises ValueError", pos_all_raised)

try:
    SPIDContext([None], vB, muB, 1.0, 1.0, 1.0, 1.0, positions="bogus")
    pos_bogus_raised = False
except ValueError:
    pos_bogus_raised = True
check("C: positions='bogus' raises ValueError", pos_bogus_raised)

layersC = [FakeBlock(torch.randn(DA, DA, generator=gA) * 0.2) for _ in range(3)]
ctxC = SPIDContext(layersC, vA[:3].clone(), muA[:3].clone(), 1.0, 1.0, 1.0, 1.0)
ctxC.__enter__()
try:
    ctxC.__enter__()
    reentry_raised = False
except RuntimeError:
    reentry_raised = True
ctxC.__exit__(None, None, None)
check("C: re-entry raises RuntimeError", reentry_raised)

baseline_C = h0A[:, :3, :DA].clone()
for layer in layersC:
    baseline_C = run_one_layer(layer, baseline_C)
after_exit_C = h0A[:, :3, :DA].clone()
for layer in layersC:
    after_exit_C = run_one_layer(layer, after_exit_C)
check("C: after __exit__ a forward is unmodified", torch.allclose(after_exit_C, baseline_C))

diagC = ctxA.diagnostics()
check("C: diagnostics() carries the standard ALQR keys + integral_norm_max",
      set(diagC) == {"steered_rows", "applied_delta_norm_mean", "applied_delta_norm_max",
                     "integral_norm_max"})
check("C: diagnostics() integral_norm_max is the max row-norm of e_sum ever applied",
      diagC["integral_norm_max"] >= float(captured_state_after_A[11][0].norm(dim=-1).max()) - 1e-4)


# ============================================================ C2. _state_for contract
# The reset triggers are (first use, batch-size change, prefill T>1) ONLY --
# a device hop at k==0 (every decode step on a sharded model: last layer's
# device != layer 0's) must MIGRATE, never reset, or e_prev is silently
# zeroed each step (= d_reset_between_steps without asking).
ctxS = SPIDContext([None] * 2, torch.randn(2, 4, generator=gA), torch.rand(2, generator=gA) + 0.5,
                   1.0, 1.0, 1.0, 1.0)
s0 = ctxS._state_for(0, torch.device("cpu"), 3, 5)
check("C2: first call initializes zeroed [B, d] state",
      all(torch.equal(t, torch.zeros(3, 4)) for t in s0))
markerS = (torch.full((3, 4), 7.0), torch.full((3, 4), 9.0))
ctxS._state = markerS
s1 = ctxS._state_for(0, torch.device("cpu"), 3, 1)
check("C2: a decode step (T=1, same B, same device) carries state -- no reset",
      s1[0] is markerS[0] and s1[1] is markerS[1])
s2 = ctxS._state_for(0, torch.device("cpu"), 3, 6)
check("C2: a new prefill (T>1) zeroes BOTH e_sum and e_prev",
      torch.equal(s2[0], torch.zeros(3, 4)) and torch.equal(s2[1], torch.zeros(3, 4)))
ctxS._state = (torch.full((3, 4), 1.0), torch.full((3, 4), 2.0))
s3 = ctxS._state_for(0, torch.device("cpu"), 2, 1)
check("C2: a batch-size change re-initializes state at the new shape",
      s3[0].shape == (2, 4) and torch.equal(s3[0], torch.zeros(2, 4)))

if torch.cuda.is_available():
    devH, devD = torch.device("cpu"), torch.device("cuda:0")
    ctxM = SPIDContext([None] * 2, torch.randn(2, 4, generator=gA),
                       torch.rand(2, generator=gA) + 0.5, 1.0, 1.0, 1.0, 1.0)
    ctxM._state_for(0, devH, 2, 4)
    ctxM._state = (torch.full((2, 4), 3.0), torch.full((2, 4), 5.0))
    sA2 = ctxM._state_for(1, devD, 2, 1)
    check("C2/gpu: mid-pass device migration preserves values",
          sA2[1].device == devD and float(sA2[1][0, 0]) == 5.0)
    sB2 = ctxM._state_for(0, devH, 2, 1)
    check("C2/gpu: a k==0 device hop at a decode step MIGRATES back (e_prev survives)",
          sB2[1].device.type == "cpu" and float(sB2[1][0, 0]) == 5.0)
else:
    print("SKIP C2 two-device leg: no CUDA")


# ============================================================ D. fit end-to-end, fake LM
tf = types.ModuleType("transformers")
tf.set_seed = lambda s: torch.manual_seed(s)
sys.modules["transformers"] = tf

DD, VD, LD_, BOS, EOS, PAD = 10, 2000, 3, 1, 2, 0


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


gD = torch.Generator().manual_seed(3)
lmD = FakeLM(DD, VD, LD_, gD)
for p in lmD.parameters():
    p.requires_grad_(False)
fake_tokD = FakeTok()
subjectD = GeneratorRef("fake/spid-fit-subject")
genD = GeneratorLM(subjectD, lmD, fake_tokD)

import polars as pl  # noqa: E402


def mk_row(concept_id, prompt, response, output_concept, category, genre="text"):
    return {
        "concept_id": concept_id,
        "prompt": prompt,
        "response": response,
        "meta": {"output_concept": output_concept, "category": category, "concept_genre": genre},
    }


rowsD = []
for i in range(6):
    rowsD.append(mk_row(0, f"tell me about A{i}", "A" * (i + 3) + "!", "conceptA", "positive"))
for i in range(6):
    rowsD.append(mk_row(1, f"tell me about B{i}", f"bland text response {i}", "conceptB", "positive"))
for i in range(8):
    rowsD.append(mk_row(-1, f"neutral prompt {i}", f"a plain neutral reply {i}", "", "negative"))
samplesD = pl.DataFrame(rowsD)
conceptsD = [Concept(id=0, name="conceptA", description=""), Concept(id=1, name="conceptB", description="")]

cfgD = SPIDConfig(subject=subjectD, batch_size=2, max_length=64, seed=7)
modelD = SPIDModel(cfgD).bind(lmD, fake_tokD)
modelD.fit(conceptsD, samplesD)

check("D: SPIDModel.fit: concept ids present", set(modelD.concepts) == {0, 1})
for cid in (0, 1):
    c = modelD.concepts[cid]
    check(f"D: SPIDModel.fit: concept {cid} v shape [{LD_},{DD}]", tuple(c["v"].shape) == (LD_, DD))
    check(f"D: SPIDModel.fit: concept {cid} mu shape [{LD_}]", tuple(c["mu"].shape) == (LD_,))
    norms_v = c["v"].norm(dim=-1)
    check(f"D: SPIDModel.fit: concept {cid} every v row unit or zero",
          bool(torch.all(((norms_v - 1.0).abs() < 1e-4) | (norms_v < 1e-6))))

pos_texts0D, neg_texts0D = _contrast_texts(samplesD, conceptsD[0], fake_tokD)
pos_X0D = _all_layer_last_token(genD, pos_texts0D, cfgD)
neg_X0D = _all_layer_last_token(genD, neg_texts0D, cfgD)
e0D = pos_X0D.mean(dim=0) - neg_X0D.mean(dim=0)
mu0D = e0D.norm(dim=-1)
zero0D = mu0D == 0
v0D = torch.zeros_like(e0D)
v0D[~zero0D] = e0D[~zero0D] / mu0D[~zero0D].unsqueeze(-1)
check("D: SPIDModel.fit: concept 0 mu matches hand-computed mean-diff norms",
      torch.allclose(modelD.concepts[0]["mu"], mu0D, atol=1e-5))
check("D: SPIDModel.fit: concept 0 v matches hand-computed unit directions",
      torch.allclose(modelD.concepts[0]["v"], v0D, atol=1e-5))

modelD_b = SPIDModel(SPIDConfig(subject=subjectD, batch_size=2, max_length=64, seed=7)).bind(lmD, fake_tokD)
modelD_b.fit(conceptsD, samplesD)
check("D: SPIDModel.fit: deterministic across two fits, same seed",
      torch.equal(modelD.concepts[0]["v"], modelD_b.concepts[0]["v"])
      and torch.equal(modelD.concepts[0]["mu"], modelD_b.concepts[0]["mu"]))

concept_no_match_D = Concept(id=5, name="no-such-concept", description="")
model_empty_pos_D = SPIDModel(cfgD).bind(lmD, fake_tokD)
try:
    model_empty_pos_D.fit([concept_no_match_D], samplesD)
    empty_pos_raised_D = False
except ValueError:
    empty_pos_raised_D = True
check("D: SPIDModel.fit: concept with zero positive rows raises ValueError", empty_pos_raised_D)

rowsD_no_neg = [r for r in rowsD if r["meta"]["category"] != "negative"]
samplesD_no_neg = pl.DataFrame(rowsD_no_neg)
model_empty_neg_D = SPIDModel(cfgD).bind(lmD, fake_tokD)
try:
    model_empty_neg_D.fit(conceptsD, samplesD_no_neg)
    empty_neg_raised_D = False
except ValueError:
    empty_neg_raised_D = True
check("D: SPIDModel.fit: concept with zero negative rows raises ValueError", empty_neg_raised_D)

ids0D = fake_tokD(pos_texts0D[:3], return_tensors="pt", padding=True)["input_ids"]
base_h4D = lmD(input_ids=ids0D).hidden_states[-1]
with modelD.context(conceptsD[0], lmbda=2.0, positions="generated") as ctxD:
    steered_h4D = lmD(input_ids=ids0D).hidden_states[-1]
check("D: SPIDModel.context: steers a real forward (final hidden state differs)",
      not torch.allclose(steered_h4D, base_h4D))
diagD = ctxD.diagnostics()
check("D: SPIDModel.context: diagnostics steered_rows == layers x batch",
      diagD["steered_rows"] == LD_ * 3)

unboundD = SPIDModel(SPIDConfig(subject=subjectD))
try:
    unboundD.context(conceptsD[0])
    not_bound_raised_D = False
except RuntimeError:
    not_bound_raised_D = True
check("D: SPIDModel.context: not bound raises RuntimeError", not_bound_raised_D)

model_mismatch_D = SPIDModel(cfgD).bind(lmD, fake_tokD)
model_mismatch_D.concepts[0] = {"v": torch.zeros(LD_ + 1, DD), "mu": torch.zeros(LD_ + 1)}
try:
    model_mismatch_D.context(conceptsD[0])
    layer_mismatch_raised_D = False
except ValueError:
    layer_mismatch_raised_D = True
check("D: SPIDModel.context: artifact layer count != bound model raises ValueError",
      layer_mismatch_raised_D)


# ============================================================ E. from_alqr
cfg_alqr_E = ALQRConfig(subject=subjectD, batch_size=2, max_length=64, jacobian_prompts=4,
                        jacobian_max_ctx=32, seed=7)
model_alqr_E = ALQRModel(cfg_alqr_E).bind(lmD, fake_tokD)
model_alqr_E.fit(conceptsD, samplesD)

spid_from_alqr = SPIDModel.from_alqr(model_alqr_E)
check("E: from_alqr: concept ids carried over", set(spid_from_alqr.concepts) == set(model_alqr_E.concepts))
check("E: from_alqr: v tensors bitwise equal to the source ALQRModel's",
      all(torch.equal(spid_from_alqr.concepts[cid]["v"], model_alqr_E.concepts[cid]["v"])
          for cid in model_alqr_E.concepts))
check("E: from_alqr: mu tensors bitwise equal to the source ALQRModel's",
      all(torch.equal(spid_from_alqr.concepts[cid]["mu"], model_alqr_E.concepts[cid]["mu"])
          for cid in model_alqr_E.concepts))
check("E: from_alqr: extraction knobs carried from the ALQR config",
      spid_from_alqr.config.subject == cfg_alqr_E.subject
      and spid_from_alqr.config.batch_size == cfg_alqr_E.batch_size
      and spid_from_alqr.config.max_length == cfg_alqr_E.max_length
      and spid_from_alqr.config.seed == cfg_alqr_E.seed)
check("E: from_alqr: default gains are the SPIDConfig constructor defaults",
      spid_from_alqr.config.kp == 10.0 and spid_from_alqr.config.ki == 10.0
      and spid_from_alqr.config.kd == 1.0)

spid_overrides = SPIDModel.from_alqr(model_alqr_E, kp=0.1, ki=0.1, kd=0.0,
                                     integral_reset_every=5, d_reset_between_steps=True)
check("E: from_alqr: overrides applied (gains)",
      spid_overrides.config.kp == 0.1 and spid_overrides.config.ki == 0.1
      and spid_overrides.config.kd == 0.0)
check("E: from_alqr: overrides applied (reset knobs)",
      spid_overrides.config.integral_reset_every == 5
      and spid_overrides.config.d_reset_between_steps is True)
check("E: from_alqr: overrides don't disturb the carried (v, mu)",
      torch.equal(spid_overrides.concepts[0]["v"], model_alqr_E.concepts[0]["v"]))

spid_bound_E = spid_from_alqr.__class__.from_alqr(model_alqr_E).bind(lmD, fake_tokD)
with spid_bound_E.context(conceptsD[0], lmbda=1.5) as ctxE:
    lmD(input_ids=ids0D)
check("E: from_alqr model steers a real forward without recomputing (v, mu)",
      ctxE.diagnostics()["steered_rows"] == LD_ * 3)


# ============================================================ F. save/load
import tempfile

SP = os.environ.get("SPID_SCRATCH") or tempfile.mkdtemp(prefix="test_spid_")
ckpt_path = f"{SP}/test_spid_ckpt.pt"
modelD.save(ckpt_path)
loadedD = SPIDModel.load(ckpt_path)
check("F: save/load roundtrip: concept ids preserved", set(loadedD.concepts) == set(modelD.concepts))
check("F: save/load roundtrip: v/mu tensors equal", all(
    torch.equal(loadedD.concepts[cid]["v"], modelD.concepts[cid]["v"])
    and torch.equal(loadedD.concepts[cid]["mu"], modelD.concepts[cid]["mu"])
    for cid in modelD.concepts))
check("F: save/load roundtrip: config fields preserved",
      loadedD.config.kp == cfgD.kp and loadedD.config.ki == cfgD.ki and loadedD.config.kd == cfgD.kd
      and loadedD.config.seed == cfgD.seed and loadedD.config.subject == cfgD.subject)

bad_cls_path = f"{SP}/test_spid_badcls.pt"
torch.save({"cls": "NotSPIDModel", "config": {}, "concepts": {}}, bad_cls_path)
try:
    SPIDModel.load(bad_cls_path)
    wrong_cls_raised = False
except ValueError:
    wrong_cls_raised = True
check("F: save/load: wrong-cls artifact raises ValueError", wrong_cls_raised)

bad_keys_path = f"{SP}/test_spid_badkeys.pt"
cfg_d_F = {**{"batch_size": 1, "max_length": 8, "seed": 1, "kp": 1.0, "ki": 1.0, "kd": 1.0,
             "integral_reset_every": 10, "d_reset_between_steps": False, "layer": None, "site": None},
           "subject": {"model_id": "x", "revision": None}}
torch.save({
    "cls": "SPIDModel", "config": cfg_d_F,
    "concepts": {0: {"v": torch.zeros(2, 3)}},  # missing "mu"
}, bad_keys_path)
try:
    SPIDModel.load(bad_keys_path)
    bad_keys_raised = False
except ValueError:
    bad_keys_raised = True
check("F: save/load: missing-buffer-key artifact raises ValueError", bad_keys_raised)

bad_lenmu_path = f"{SP}/test_spid_badlenmu.pt"
torch.save({
    "cls": "SPIDModel", "config": cfg_d_F,
    "concepts": {0: {"v": torch.zeros(2, 3), "mu": torch.zeros(3)}},  # len(mu) != v.shape[0]
}, bad_lenmu_path)
try:
    SPIDModel.load(bad_lenmu_path)
    bad_lenmu_raised = False
except ValueError:
    bad_lenmu_raised = True
check("F: save/load: mu/v length mismatch raises ValueError", bad_lenmu_raised)


# ============================================================ G. real-HF leg (tiny random Llama, offline)
if _real_transformers is None:
    print("SKIP real-HF leg: transformers unavailable")
else:
    HF_SEED = 123
    HF_VOCAB = 300
    HF_D = 32
    HF_LAYERS = 2
    HF_HEADS = 4
    HF_KV_HEADS = 2

    def _mk_real_model():
        torch.manual_seed(HF_SEED)
        cfg = _RealLlamaConfig(vocab_size=HF_VOCAB, hidden_size=HF_D, intermediate_size=2 * HF_D,
                               num_hidden_layers=HF_LAYERS, num_attention_heads=HF_HEADS,
                               num_key_value_heads=HF_KV_HEADS, max_position_embeddings=64,
                               pad_token_id=PAD, attn_implementation="eager")
        m = _RealLlamaForCausalLM(cfg).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    llamaG = _mk_real_model()

    # ---- G1. e_prev cross-decode-step leak, exercised under real KV-cache manual decode
    def _manual_steps(model, ctx, prompt_ids, step_ids):
        layers = model.model.layers
        captures = {k: [] for k in range(len(layers))}

        def mk_cap(k):
            def cap(mod, args, output):
                h = output[0] if isinstance(output, tuple) else output
                captures[k].append((args[0].detach().clone(), h.detach().clone()))
            return cap

        if ctx is not None:
            ctx.__enter__()
        caps = [lay.register_forward_hook(mk_cap(k)) for k, lay in enumerate(layers)]
        past = None
        attn_len = prompt_ids.shape[1]
        with torch.no_grad():
            out = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                       use_cache=True, past_key_values=past)
            past = out.past_key_values
            for tok in step_ids:
                next_ids = torch.tensor([[tok]], dtype=torch.long)
                attn_len += 1
                out = model(input_ids=next_ids,
                           attention_mask=torch.ones(1, attn_len, dtype=torch.long),
                           use_cache=True, past_key_values=past)
                past = out.past_key_values
        for c in caps:
            c.remove()
        if ctx is not None:
            ctx.__exit__(None, None, None)
        return captures

    gG = torch.Generator().manual_seed(44)
    L_g = len(llamaG.model.layers)
    D_g = llamaG.config.hidden_size
    vG = torch.randn(L_g, D_g, generator=gG)
    vG = vG / vG.norm(dim=-1, keepdim=True)
    muG = torch.rand(L_g, generator=gG) + 0.5
    kpG, kiG, kdG = 1.7, 0.9, 0.6
    lmbdaG = 1.2
    prompt_idsG = torch.randint(3, HF_VOCAB, (1, 6), generator=gG)
    step_idsG = [int(x) for x in torch.randint(3, HF_VOCAB, (3,), generator=gG)]
    num_calls_G = 1 + len(step_idsG)

    ctx_leak_G = SPIDContext(llamaG.model.layers, vG.clone(), muG.clone(), kpG, kiG, kdG,
                             lmbdaG, integral_reset_every=10, d_reset_between_steps=False)
    cap_leak_G = _manual_steps(llamaG, ctx_leak_G, prompt_idsG, step_idsG)

    ctx_reset_G = SPIDContext(llamaG.model.layers, vG.clone(), muG.clone(), kpG, kiG, kdG,
                              lmbdaG, integral_reset_every=10, d_reset_between_steps=True)
    cap_reset_G = _manual_steps(llamaG, ctx_reset_G, prompt_idsG, step_idsG)

    cap_base_G = _manual_steps(llamaG, None, prompt_idsG, step_idsG)

    def e_at(cap, k, c):
        x_in = cap[k][c][0]
        x_last = x_in[:, -1, :].float()
        beta_k = lmbdaG * muG[k].item()
        alpha = beta_k - x_last @ vG[k]
        return alpha[:, None] * vG[k]

    leak_ok = True
    reset_ok = True
    for c in range(1, num_calls_G):
        e0c = e_at(cap_leak_G, 0, c)  # layer 0's x_in is identical across all three runs
        e_prev_leak = e_at(cap_leak_G, L_g - 1, c - 1)  # last layer's e from the PREVIOUS call
        u0_leak_expected = kpG * e0c + kdG * (e0c - e_prev_leak)  # layer 0 always resets the integral
        delta0_leak = (cap_leak_G[0][c][1][:, -1, :] - cap_base_G[0][c][1][:, -1, :]).float()
        leak_ok &= bool(torch.allclose(u0_leak_expected, delta0_leak, atol=5e-4))

        u0_reset_expected = kpG * e0c + kdG * e0c  # e_prev forced to zero
        delta0_reset = (cap_reset_G[0][c][1][:, -1, :] - cap_base_G[0][c][1][:, -1, :]).float()
        reset_ok &= bool(torch.allclose(u0_reset_expected, delta0_reset, atol=5e-4))
    check("G1: e_prev LEAKS across decode steps (d_reset_between_steps=False, real KV-cache)",
          leak_ok)
    check("G1: d_reset_between_steps=True zeroes e_prev at each new decode step",
          reset_ok)
    check("G1: the leak and reset variants actually differ (sanity: test isn't vacuous)",
          any(not torch.allclose(cap_leak_G[0][c][1], cap_reset_G[0][c][1], atol=1e-6)
              for c in range(1, num_calls_G)))

    # ---- G2. prefill detection + batch-size change, via real .generate()
    gG2 = torch.Generator().manual_seed(45)
    vG2 = torch.randn(L_g, D_g, generator=gG2)
    vG2 = vG2 / vG2.norm(dim=-1, keepdim=True)
    muG2 = torch.rand(L_g, generator=gG2) + 0.3
    kpG2, kiG2, kdG2 = 0.8, 0.4, 0.3
    lmbdaG2 = 1.1

    def _spid_ctx_G2():
        return SPIDContext(llamaG.model.layers, vG2.clone(), muG2.clone(), kpG2, kiG2, kdG2,
                           lmbdaG2, integral_reset_every=10)

    def _gen(ctx, ids):
        with ctx, torch.no_grad():
            return llamaG.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=4, min_new_tokens=4, do_sample=False,
                                   pad_token_id=0)

    ids1_G2 = torch.randint(3, HF_VOCAB, (1, 6), generator=gG2)
    ids2_G2 = torch.randint(3, HF_VOCAB, (1, 5), generator=gG2)

    shared_ctx_G2 = _spid_ctx_G2()
    out1_shared = _gen(shared_ctx_G2, ids1_G2)
    out2_shared = _gen(shared_ctx_G2, ids2_G2)
    out1_fresh = _gen(_spid_ctx_G2(), ids1_G2)
    out2_fresh = _gen(_spid_ctx_G2(), ids2_G2)
    check("G2: prefill detection -- shared-context sequential calls == fresh context per call (1)",
          torch.equal(out1_shared, out1_fresh))
    check("G2: prefill detection -- shared-context sequential calls == fresh context per call (2)",
          torch.equal(out2_shared, out2_fresh))

    ids3_G2 = torch.randint(3, HF_VOCAB, (3, 5), generator=gG2)
    crash_G2 = False
    out3_shared = None
    try:
        out3_shared = _gen(shared_ctx_G2, ids3_G2)
    except Exception:
        crash_G2 = True
    check("G2: a batch-size change under a shared context doesn't crash", not crash_G2)
    if not crash_G2:
        out3_fresh = _gen(_spid_ctx_G2(), ids3_G2)
        check("G2: batch-size change: result matches a fresh context (state correctly reset)",
              torch.equal(out3_shared, out3_fresh))

    # direct state pins: greedy-token equality on this tiny model is too weak
    # to catch a broken prefill detector (verified by mutation -- removing the
    # T>1 trigger left every output-equality check green)
    ctx_state_G2 = _spid_ctx_G2()
    _gen(ctx_state_G2, ids1_G2)
    carried_G2 = ctx_state_G2._state
    check("G2 state pin: after a generate, carried state is NONZERO (decode accumulated)",
          carried_G2 is not None and bool(carried_G2[1].abs().sum() > 0))
    s_decode_G2 = ctx_state_G2._state_for(0, carried_G2[0].device, carried_G2[0].shape[0], 1)
    check("G2 state pin: a T=1 same-B call returns the CARRIED state (leak preserved)",
          s_decode_G2[0] is carried_G2[0] and s_decode_G2[1] is carried_G2[1])
    s_prefill_G2 = ctx_state_G2._state_for(0, carried_G2[0].device,
                                           ids1_G2.shape[0], ids1_G2.shape[1])
    check("G2 state pin: the next prefill (T>1, same B) returns ZEROED state",
          bool(s_prefill_G2[0].abs().sum() == 0) and bool(s_prefill_G2[1].abs().sum() == 0))


# ============================================================ H. ALQR_DIR-gated upstream parity leg
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
        import steer.PIDsteering as upstream_pid
    except Exception as exc:
        print(f"SKIP upstream source-parity leg: import failed ({exc!r})")
        upstream_pid = None

    if upstream_pid is not None:
        gH = torch.Generator().manual_seed(55)
        Lp, n, Bp = 12, 8, 3
        E_H = torch.randn(Lp, n, generator=gH)
        mu_H = E_H.norm(dim=-1)
        v_H = E_H / mu_H.unsqueeze(-1)
        lmbda_H = 1.6
        num_passes_H = 2

        pregen_H = [[(torch.randn(Bp, 1, n, generator=gH), (torch.randn(Bp, 1, n, generator=gH),))
                    for _ in range(Lp)] for _ in range(num_passes_H)]

        for gains_name, (kp_H, ki_H, kd_H) in (
            ("constructor defaults (10, 10, 1)", (10.0, 10.0, 1.0)),
            ("llama TQA table (0.1, 0.1, 0.0)", (0.1, 0.1, 0.0)),
        ):
            lqr_obj = object.__new__(upstream_pid.PIDSteering)
            lqr_obj.device = torch.device("cpu")
            lqr_obj.Kp, lqr_obj.Ki, lqr_obj.Kd = kp_H, ki_H, kd_H
            lqr_obj.E = E_H
            lqr_obj.E_unit = v_H.clone()
            lqr_obj.betas = [lmbda_H * float(m) for m in mu_H]
            lqr_obj.e_sum = torch.zeros(Bp, n)
            lqr_obj.e_prev = torch.zeros(Bp, n)
            lqr_obj.X = torch.zeros(Lp + 1, n)
            lqr_obj.U = torch.zeros(Lp, n)

            ctx_H = SPIDContext([None] * Lp, v_H.clone(), mu_H.clone(), kp_H, ki_H, kd_H,
                                lmbda_H, integral_reset_every=10)

            parity_ok = True
            for p in range(num_passes_H):
                for k in range(Lp):
                    x_in_pk, fake_out_pk = pregen_H[p][k]
                    original_out = fake_out_pk[0].clone()
                    upstream_out = lqr_obj.hook_setpoint_tracking(
                        k, None, (x_in_pk,), (fake_out_pk[0].clone(),))
                    upstream_delta = (upstream_out[0] - original_out)[:, -1, :]

                    our_out = ctx_H._apply(k, x_in_pk, fake_out_pk[0].clone())
                    our_delta = (our_out - original_out)[:, -1, :]
                    parity_ok &= bool(torch.allclose(our_delta, upstream_delta, atol=1e-5))
            check(f"H: SPIDContext matches hook_setpoint_tracking per-layer, multi-pass, "
                  f"gains={gains_name}", parity_ok)


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
