#!/usr/bin/env python3
# Offline tests for steergym.flas (CachedFLASSteerModel): the registry/
# mismatch validation, the CachedFLASContext Euler-integration hook (toy
# field, padding invariance, cached-vs-recompute decoding equality, dtype
# round-trip), and the ported FlowBlock/FlowFunction/ConceptEncoder
# architecture. Needs CPU torch + transformers. Run:
#   PYTHONPATH=<pylibs>:<torchlibs>:<flaslibs> python3 steergym/tests/test_cached_flas.py
# Two gated legs (skip cleanly without their resource):
#   FLAS_DIR=<clone of github.com/flas-ai/FLAS> -- upstream source parity
#   network/HF cache reachable -- real flas-gemma-2-2b-it strict state_dict load
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from steergym.concept import Concept
from steergym.flas import flow_model as flas_flow_model
from steergym.flas.models import (
    FLAS_CHECKPOINTS,
    CachedFLASConfig,
    CachedFLASContext,
    CachedFLASSteerModel,
    FLASCheckpointError,
    _padding_mask_from_attn_mask,
)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


def _tiny_gemma2_config(**overrides):
    from transformers.models.gemma2.configuration_gemma2 import Gemma2Config

    kwargs = dict(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                 max_position_embeddings=64, sliding_window=4096, attn_logit_softcapping=None,
                 final_logit_softcapping=None, attn_implementation="eager", pad_token_id=0)
    kwargs.update(overrides)
    return Gemma2Config(**kwargs)


def _tiny_gemma2_model(seed=0):
    from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM

    cfg = _tiny_gemma2_config()
    torch.manual_seed(seed)
    model = Gemma2ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=64):
        n = min(max_length, max(2, len(text) % 5 + 2))
        ids = torch.arange(1, n + 1) % 49 + 1
        return {"input_ids": ids.unsqueeze(0), "attention_mask": torch.ones(1, n)}


class _FakeConfig:
    def __init__(self, model_type, hidden_size):
        self.model_type = model_type
        self.hidden_size = hidden_size


class _FakeSubject(nn.Module):
    def __init__(self, model_type, hidden_size, n_layers=21):
        super().__init__()
        self.config = _FakeConfig(model_type, hidden_size)
        inner = nn.Module()
        inner.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.model = inner

    @property
    def device(self):
        return next(self.parameters()).device


# ================================================================== 1. registry validation
try:
    CachedFLASSteerModel.from_pretrained("not-a-real/flas-repo")
    unknown_raised = False
except FLASCheckpointError:
    unknown_raised = True
check("from_pretrained: unknown repo raises FLASCheckpointError (1)", unknown_raised)

check("registry: every entry has the expected fields (1)",
      all({"model_id", "model_type", "hidden_size", "layer", "num_blocks", "n_steps",
          "disable_self_attn", "prompt_format"} <= set(v) for v in FLAS_CHECKPOINTS.values()))

m2b = CachedFLASSteerModel.from_pretrained("flas-ai/flas-gemma-2-2b-it")
check("from_pretrained: config.hf_repo set (1)", m2b.config.hf_repo == "flas-ai/flas-gemma-2-2b-it")
check("from_pretrained: config.subject.model_id from registry (1)",
      m2b.config.subject.model_id == "google/gemma-2-2b-it")
check("from_pretrained: config.layer from registry (1)", m2b.config.layer == 20)
check("from_pretrained: config.model_type from registry (1)", m2b.config.model_type == "gemma2")
check("from_pretrained: config.hidden_size from registry (1)", m2b.config.hidden_size == 2304)
check("from_pretrained: config.disable_self_attn from registry (1)", m2b.config.disable_self_attn is False)


# ================================================================== 2. mismatch refusal at bind()
wrong_type_subject = _FakeSubject("qwen3", 2304)
m = CachedFLASSteerModel.from_pretrained("flas-ai/flas-gemma-2-2b-it")
try:
    m.bind(wrong_type_subject, _FakeTokenizer())
    raised = False
except FLASCheckpointError:
    raised = True
check("bind: wrong model_type refused (2)", raised)

wrong_size_subject = _FakeSubject("gemma2", 999)
m = CachedFLASSteerModel.from_pretrained("flas-ai/flas-gemma-2-2b-it")
try:
    m.bind(wrong_size_subject, _FakeTokenizer())
    raised = False
except FLASCheckpointError:
    raised = True
check("bind: wrong hidden_size refused (2)", raised)

right_subject = _FakeSubject("gemma2", 2304)
m = CachedFLASSteerModel.from_pretrained("flas-ai/flas-gemma-2-2b-it")
try:
    m.bind(right_subject, _FakeTokenizer())
    bound_ok = True
except FLASCheckpointError:
    bound_ok = False
check("bind: matching model_type + hidden_size accepted (2)", bound_ok)


check("_padding_mask_from_attn_mask: None passthrough (2)",
      _padding_mask_from_attn_mask(None) is None)
mask4d = torch.tensor([[[[float("-inf"), 0.0, 0.0]]]])
check("_padding_mask_from_attn_mask: 4D additive mask, last row (2)",
      torch.equal(_padding_mask_from_attn_mask(mask4d), torch.tensor([[0.0, 1.0, 1.0]])))
mask2d_bool = torch.tensor([[False, True, True]])
check("_padding_mask_from_attn_mask: 2D boolean mask (2)",
      torch.equal(_padding_mask_from_attn_mask(mask2d_bool), torch.tensor([[0.0, 1.0, 1.0]])))


def _raises(fn, exc_type):
    try:
        fn()
        return False
    except exc_type:
        return True


# ================================================================== 3. Euler integration vs hand-computed toy field
class _ToyFlow:
    """v(h, t, c) = alpha*h, independent of t/c -- closed-form Euler:
    h_n = h_0 * (1 + dt*alpha)**n_steps."""

    def __init__(self, alpha, dtype=torch.float32):
        self._p = nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        self.alpha = alpha

    def parameters(self):
        return iter([self._p])

    def __call__(self, h, concept_hidden, concept_mask, t=None, self_attn_caches=None,
                 padding_mask=None, use_cache=False, past_len=0, position_ids=None):
        return self.alpha * h, [None]


toy = _ToyFlow(alpha=0.3)
h0 = torch.randn(2, 3, 16)
dummy_c = torch.zeros(1, 1, 16)
dummy_mask = torch.ones(1, 1)
T, n_steps = 0.9, 4
ctx = CachedFLASContext(None, toy, dummy_c, dummy_mask, T=T, n_steps=n_steps)
h_out = ctx._step(h0, None, torch.arange(3).unsqueeze(0))
dt = T / n_steps
expected = h0 * (1 + dt * toy.alpha) ** n_steps
check("toy field: Euler integration matches closed form (3)",
      torch.allclose(h_out, expected, rtol=1e-5, atol=1e-6))

ctx0 = CachedFLASContext(None, toy, dummy_c, dummy_mask, T=0.0, n_steps=n_steps)
h_out0 = ctx0._step(h0, None, torch.arange(3).unsqueeze(0))
check("toy field: T=0 is exact identity (3)", torch.equal(h_out0, h0))

check("CachedFLASContext: n_steps<=0 raises ValueError (3)",
      _raises(lambda: CachedFLASContext(None, toy, dummy_c, dummy_mask, T=1.0, n_steps=0), ValueError))


# ================================================================== 4. real FlowFunction: T=0 identity + diagnostics
model, cfg = _tiny_gemma2_model(seed=1)
flow_fn = flas_flow_model.FlowFunction(cfg, num_blocks=1, time_conditioned=True, layer_idx=1,
                                  disable_cross_attn=False, disable_self_attn=False, disable_mlp=False)
for p in flow_fn.parameters():
    p.requires_grad = False
concept_enc = flas_flow_model.ConceptEncoder.from_base_model(model, num_layers=2)
concept_ids = torch.tensor([[3, 4, 5, 6]])
concept_mask = torch.ones_like(concept_ids).float()
concept_hidden = concept_enc(concept_ids, concept_mask)
layer_module = model.model.layers[1]

ctx_id = CachedFLASContext(layer_module, flow_fn, concept_hidden, concept_mask, T=0.0, n_steps=3)
h_real = torch.randn(1, 5, 16)
out_id = ctx_id._step(h_real, None, torch.arange(5).unsqueeze(0))
check("real FlowFunction: T=0 identity (4)", torch.equal(out_id, h_real))
check("real FlowFunction: T=0 diagnostics report zero applied delta (4)",
      ctx_id.diagnostics()["applied_delta_norm_mean"] == 0.0)

ctx_nz = CachedFLASContext(layer_module, flow_fn, concept_hidden, concept_mask, T=1.3, n_steps=3)
out_nz = ctx_nz._step(h_real, None, torch.arange(5).unsqueeze(0))
diag = ctx_nz.diagnostics()
check("real FlowFunction: T>0 changes the activations (4)", not torch.equal(out_nz, h_real))
check("real FlowFunction: T>0 diagnostics report nonzero applied delta (4)",
      diag["applied_delta_norm_mean"] > 0.0 and diag["steered_rows"] == 5)


# ================================================================== 5. steered-positions masking/padding invariance
def _run_capture(ids, attn, T=1.1, n_steps=3, n_new=2):
    h_outs = []

    def cap(mod, a, k, o):
        h_outs.append((o[0] if isinstance(o, tuple) else o).detach().clone())

    ctx = CachedFLASContext(layer_module, flow_fn, concept_hidden, concept_mask, T=T, n_steps=n_steps)
    with ctx:
        handle = layer_module.register_forward_hook(cap, with_kwargs=True)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=n_new,
                                 do_sample=False, use_cache=True)
        handle.remove()
    return out, h_outs


ids_alone = torch.tensor([[7, 8, 9, 10, 11]])
attn_alone = torch.ones_like(ids_alone)
out_alone, h_alone = _run_capture(ids_alone, attn_alone)

pad = 2
ids_short = torch.tensor([[20, 21, 22]])
ids_batch = torch.cat([torch.cat([torch.zeros(1, pad, dtype=torch.long), ids_short], dim=1), ids_alone], dim=0)
attn_batch = torch.cat([
    torch.cat([torch.zeros(1, pad, dtype=torch.long), torch.ones(1, 3, dtype=torch.long)], dim=1),
    attn_alone,
], dim=0)
out_batch, h_batch = _run_capture(ids_batch, attn_batch)

check("padding invariance: generated tokens match alone vs left-padded batch (5)",
      torch.equal(out_alone[0], out_batch[1]))
check("padding invariance: steered hidden states match at every forward call (5)",
      all(torch.allclose(ha, hb[1:2], atol=1e-5) for ha, hb in zip(h_alone, h_batch)))


# ================================================================== 6. cached-vs-recompute decoding equality
ids_c = torch.tensor([[7, 8, 9, 10, 11]])
attn_c = torch.ones_like(ids_c)
T_c, n_steps_c = 1.3, 3

h_origs = []


def cap_orig(mod, a, k, o):
    h_origs.append((o[0] if isinstance(o, tuple) else o).detach().clone())


cap_handle = layer_module.register_forward_hook(cap_orig, with_kwargs=True)
ctx_c = CachedFLASContext(layer_module, flow_fn, concept_hidden, concept_mask, T=T_c, n_steps=n_steps_c)
h_steered = []
with ctx_c:
    def cap_steered(mod, a, k, o):
        h_steered.append((o[0] if isinstance(o, tuple) else o).detach().clone())
    steered_handle = layer_module.register_forward_hook(cap_steered, with_kwargs=True)
    with torch.no_grad():
        model.generate(input_ids=ids_c, attention_mask=attn_c, max_new_tokens=3,
                       do_sample=False, use_cache=True)
    steered_handle.remove()
cap_handle.remove()

full_h_orig = torch.cat(h_origs, dim=1)
total_len = full_h_orig.shape[1]
dt_c = T_c / n_steps_c
h_recompute = full_h_orig.clone()
with torch.no_grad():
    for k in range(n_steps_c):
        t_k = torch.full((1,), dt_c * k)
        v, _ = flow_fn(h_recompute, concept_hidden.expand(1, -1, -1), concept_mask.expand(1, -1), t=t_k,
                       self_attn_caches=None, use_cache=False, padding_mask=None, past_len=0,
                       position_ids=torch.arange(total_len).unsqueeze(0))
        h_recompute = h_recompute + dt_c * v

check("cached-vs-recompute: last decode position matches full no-cache recompute (6)",
      torch.allclose(h_steered[-1][:, -1, :], h_recompute[:, -1, :], atol=1e-5))


# ================================================================== 7. dtype round-trip
model64, cfg64 = _tiny_gemma2_model(seed=2)
flow_fn64 = flas_flow_model.FlowFunction(cfg64, num_blocks=1, time_conditioned=True, layer_idx=1,
                                    disable_cross_attn=False, disable_self_attn=False, disable_mlp=False)
flow_fn64 = flow_fn64.double()
concept_enc64 = flas_flow_model.ConceptEncoder.from_base_model(model64, num_layers=2)
ch64 = concept_enc64(concept_ids, concept_mask).double()
ctx_dtype = CachedFLASContext(model64.model.layers[1], flow_fn64, ch64, concept_mask.double(), T=1.0, n_steps=2)
h32 = torch.randn(1, 4, 16, dtype=torch.float32)
out_dtype = ctx_dtype._step(h32, None, torch.arange(4).unsqueeze(0))
check("dtype round-trip: float32 stream through float64 flow weights returns float32 (7)",
      out_dtype.dtype == torch.float32)
check("dtype round-trip: output changed (flow actually ran) (7)", not torch.equal(out_dtype, h32))


# ================================================================== 8. FLAS_DIR-gated upstream source parity
FLAS_DIR = __import__("os").environ.get("FLAS_DIR")
if not FLAS_DIR:
    print("SKIP upstream source-parity leg: FLAS_DIR not set")
else:
    sys.path.insert(0, str(Path(FLAS_DIR) / "src"))
    try:
        import flas.model as upstream_model
    except Exception as exc:  # pytorch_lightning / missing deps etc.
        print(f"SKIP upstream source-parity leg: import failed ({exc!r})")
        upstream_model = None

    if upstream_model is not None:
        p_cfg = _tiny_gemma2_config()
        torch.manual_seed(11)
        theirs = upstream_model.FlowFunction(p_cfg, num_blocks=1, time_conditioned=True, layer_idx=1,
                                             disable_cross_attn=False, disable_self_attn=False, disable_mlp=False)
        torch.manual_seed(11)
        mine = flas_flow_model.FlowFunction(p_cfg, num_blocks=1, time_conditioned=True, layer_idx=1,
                                       disable_cross_attn=False, disable_self_attn=False, disable_mlp=False)
        same_init = (set(theirs.state_dict()) == set(mine.state_dict())
                    and all(torch.equal(theirs.state_dict()[k], mine.state_dict()[k]) for k in theirs.state_dict()))
        check("upstream parity: identical module order/param count under the same seed (8)", same_init)

        hp = torch.randn(1, 5, 16)
        cp = torch.randn(1, 4, 16)
        cm = torch.ones(1, 4)
        posp = torch.arange(5).unsqueeze(0)
        with torch.no_grad():
            v_theirs, _ = theirs(hp, cp, cm, t=torch.zeros(1), use_cache=False, position_ids=posp)
            v_mine, _ = mine(hp, cp, cm, t=torch.zeros(1), use_cache=False, position_ids=posp)
        check("upstream parity: velocity field bit-exact (8)", torch.equal(v_theirs, v_mine))

        Tp, nstepsp = 1.7, 3
        ctx_p = CachedFLASContext(None, mine, cp, cm, T=Tp, n_steps=nstepsp)
        ours_out = ctx_p._step(hp, None, posp)
        dtp = Tp / nstepsp
        h2 = hp.clone()
        with torch.no_grad():
            for k in range(nstepsp):
                v, _ = theirs(h2, cp, cm, t=torch.full((1,), dtp * k), use_cache=False, position_ids=posp)
                h2 = h2 + dtp * v
        check("upstream parity: integrated Euler output bit-exact vs hand-rolled upstream loop (8)",
              torch.equal(ours_out, h2))


# ================================================================== 9. real-checkpoint strict state_dict leg
try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except ImportError as exc:
    hf_hub_download = None
    print(f"SKIP real-checkpoint strict-load leg: huggingface_hub/safetensors unavailable ({exc!r})")

if hf_hub_download is not None:
    try:
        path = hf_hub_download("flas-ai/flas-gemma-2-2b-it", "flas-gemma-2-2b-it.safetensors")
        state_dict = load_file(path, device="cpu")
    except Exception as exc:
        print(f"SKIP real-checkpoint strict-load leg: fetch failed ({exc!r})")
        state_dict = None

    if state_dict is not None:
        from transformers.models.gemma2.configuration_gemma2 import Gemma2Config

        # Gemma2Config() defaults ARE google/gemma-2-2b-it's architecture
        # (cross-checked against this checkpoint's own tensor shapes --
        # q_proj [2048, 2304], k_proj [1024, 2304] -- so no network call to
        # the (gated) google/gemma-2-2b-it repo is needed for this leg).
        real_cfg = Gemma2Config()
        entry = FLAS_CHECKPOINTS["flas-ai/flas-gemma-2-2b-it"]
        real_flow_fn = flas_flow_model.FlowFunction(
            real_cfg, num_blocks=entry["num_blocks"], time_conditioned=True, layer_idx=entry["layer"],
            disable_self_attn=entry["disable_self_attn"])
        target_dtype = next(iter(state_dict.values())).dtype
        real_flow_fn.to(target_dtype)
        try:
            missing, unexpected = real_flow_fn.load_state_dict(state_dict, strict=True)
            strict_ok = not missing and not unexpected
        except RuntimeError:
            strict_ok = False
        check("real checkpoint: flas-gemma-2-2b-it loads strict=True with zero missing/unexpected keys (9)",
              strict_ok)
        n_params = sum(p.numel() for p in real_flow_fn.parameters())
        check("real checkpoint: param count in the paper's ~100-255M FlowBlock range (9)",
              1e7 < n_params < 3e8)


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
