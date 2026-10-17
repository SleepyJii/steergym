#!/usr/bin/env python3
# Offline tests for steergym.gcad (GCAD): the RoPE/repeat_kv math oracles
# (rope.py), the Eq. 9/10 gate formula, GCADContext's constructor
# validation, and -- gated on `transformers` being importable (tiny
# random Qwen2/Llama configs, no network/weights download, same precedent
# as steergym.alqr's real-HF leg) -- the Eq. 6/7 exact-decomposition
# oracle, extraction shape/value checks, context injection (degenerate
# limits + positions), KV-cache generate equivalence, fit/save/load, and
# the attn_implementation="eager" requirement. A `GCAD_DIR`-gated leg
# replays the real upstream repo's own functions against ours on
# identical tensors; skips cleanly if the clone or its deps are absent.
# Run:
#   PYTHONPATH=<pylibs>:<torchlibs>[:<gcadlibs>] python3 steergym/tests/test_gcad.py
import os
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from steergym.concept import Concept
from steergym.gcad.models import GCADConfig, GCADContext, GCADModel
from steergym.gcad.rope import apply_rotary_pos_emb, repeat_kv, rotate_half
from steergym.generator import GeneratorRef

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


SUBJECT = GeneratorRef("fake/gcad-subject")


# ================================================================== 1. RoPE / repeat_kv math oracles
def naive_apply_rotary_pos_emb(q, k, cos, sin):
    cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)

    def rot(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    return q * cos_u + rot(q) * sin_u, k * cos_u + rot(k) * sin_u


g1 = torch.Generator().manual_seed(0)
B1, H1, T1, D1 = 2, 4, 5, 8
q1 = torch.randn(B1, H1, T1, D1, generator=g1)
k1 = torch.randn(B1, H1, T1, D1, generator=g1)
cos1 = torch.randn(B1, T1, D1, generator=g1)
sin1 = torch.randn(B1, T1, D1, generator=g1)

q_ours, k_ours = apply_rotary_pos_emb(q1, k1, cos1, sin1)
q_naive, k_naive = naive_apply_rotary_pos_emb(q1, k1, cos1, sin1)
check("apply_rotary_pos_emb matches a from-scratch rotate-half reference (1)",
      torch.equal(q_ours, q_naive) and torch.equal(k_ours, k_naive))
check("rotate_half matches the inline rot() used above (1)",
      torch.equal(rotate_half(q1), torch.cat((-q1[..., D1 // 2:], q1[..., :D1 // 2]), dim=-1)))

kv1 = torch.randn(2, 2, 6, 3, generator=g1)
rep1 = repeat_kv(kv1, 3)
check("repeat_kv shape (1)", tuple(rep1.shape) == (2, 6, 6, 3))
check("repeat_kv: output head h maps to kv head h // n_rep (1)",
      all(torch.equal(rep1[:, h], kv1[:, h // 3]) for h in range(6)))
check("repeat_kv n_rep=1 is a no-op (identity object) (1)", repeat_kv(kv1, 1) is kv1)


# ================================================================== 2. gate formula oracle (Eq. 9/10)
def naive_gate(q_rope, k_pos_per_head, avg_demand, base_coef, scale, head_dim):
    # q_rope, k_pos_per_head: [B, T, H, D] / [H, D]
    dot = (q_rope * k_pos_per_head).sum(dim=-1)        # [B, T, H]
    demand = dot.mean(dim=-1) / (head_dim ** 0.5)      # [B, T]
    centered = demand - avg_demand
    return 2.0 * base_coef * torch.sigmoid(scale * centered), demand


g2 = torch.Generator().manual_seed(1)
Bg, Tg, Hg, Dg = 2, 4, 4, 3
q_rope2 = torch.randn(Bg, Tg, Hg, Dg, generator=g2)
k_pos2 = torch.randn(Hg, Dg, generator=g2)
coef_naive, demand_naive = naive_gate(q_rope2, k_pos2, avg_demand=0.37, base_coef=2.5,
                                      scale=1.5, head_dim=Dg)
dot2 = (q_rope2 * k_pos2).sum(dim=-1)
demand2 = dot2.mean(dim=-1) / (Dg ** 0.5)
coef2 = 2.0 * 2.5 * torch.sigmoid(1.5 * (demand2 - 0.37))
check("Eq. 9 demand formula is a self-consistent reduction (2)",
      torch.allclose(demand2, demand_naive))
check("Eq. 10 gate formula is a self-consistent reduction (2)", torch.allclose(coef2, coef_naive))
zero_coef = 2.0 * 0.0 * torch.sigmoid(1.5 * (demand2 - 0.37))
check("Eq. 10: base_coef=0 is an EXACT zero gate for any demand/avg_demand/scale (2)",
      torch.equal(zero_coef, torch.zeros_like(zero_coef)))


# ================================================================== 3. GCADContext constructor validation
class _DummyLayer:
    pass


try:
    GCADContext([_DummyLayer()], [0, 1], torch.zeros(1, 4), torch.zeros(1, 2, 2), torch.zeros(1),
               num_heads=4, num_kv_heads=2, head_dim=2, base_coef=1.0, scale=1.0)
    check("GCADContext rejects mismatched layer/row counts (3)", False)
except ValueError:
    check("GCADContext rejects mismatched layer/row counts (3)", True)

try:
    GCADContext([_DummyLayer()], [0], torch.zeros(1, 4), torch.zeros(1, 2, 2), torch.zeros(1),
               num_heads=4, num_kv_heads=2, head_dim=2, base_coef=1.0, scale=1.0,
               positions="bogus")
    check("GCADContext rejects an invalid positions value (3)", False)
except ValueError:
    check("GCADContext rejects an invalid positions value (3)", True)


# ================================================================== 3b. crop window: instruction span before meta["question"]
from steergym.gcad.extract import crop_length


class _OffsetTok:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans, pos = [], 0
        for word in text.split(" "):
            spans.append((pos, pos + len(word)))
            pos += len(word) + 1
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


_rendered = "<u> You are an evil assistant. Be evil. what is love <a>"
check("crop_length stops at the question's first token (3b)",
      crop_length(_OffsetTok(), _rendered, 12, "what is love") == 8)
check("crop_length falls back to the whole prompt without a question (3b)",
      crop_length(_OffsetTok(), _rendered, 12, None) == 12)
check("crop_length falls back when the question is absent from the prompt (3b)",
      crop_length(_OffsetTok(), _rendered, 12, "not here") == 12)


# ================================================================== 4. real-HF-layer leg (tiny random Qwen2/Llama, offline)
try:
    import transformers as _real_transformers
    _Qwen2Config = _real_transformers.Qwen2Config
    _Qwen2ForCausalLM = _real_transformers.Qwen2ForCausalLM
    _LlamaConfig = _real_transformers.LlamaConfig
    _LlamaForCausalLM = _real_transformers.LlamaForCausalLM
except ImportError:
    _real_transformers = None

if _real_transformers is None:
    print("SKIP real-HF leg: transformers unavailable")
else:
    from steergym.gcad.extract import contrast_rows, cropped_attn_response_avg, k_pos_and_avg_demand, layer_dims

    HF_VOCAB, HF_D, HF_LAYERS, HF_HEADS, HF_KV_HEADS = 300, 16, 2, 4, 2

    def _mk_model(cfg_cls, model_cls, seed, **extra):
        torch.manual_seed(seed)
        cfg = cfg_cls(vocab_size=HF_VOCAB, hidden_size=HF_D, intermediate_size=2 * HF_D,
                      num_hidden_layers=HF_LAYERS, num_attention_heads=HF_HEADS,
                      num_key_value_heads=HF_KV_HEADS, max_position_embeddings=64,
                      pad_token_id=0, attn_implementation="eager", **extra)
        m = model_cls(cfg).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    MODELS = {
        "qwen2": _mk_model(_Qwen2Config, _Qwen2ForCausalLM, 10),
        "llama": _mk_model(_LlamaConfig, _LlamaForCausalLM, 10),
        # head_dim != hidden_size / heads, as on gemma-2-2b (256 vs 2304/8)
        "gemma2": _mk_model(_real_transformers.Gemma2Config, _real_transformers.Gemma2ForCausalLM, 10,
                            head_dim=8, query_pre_attn_scalar=8),
    }

    # ---- 4a. Eq. 6/7 exact-decomposition oracle: cropping to the FULL
    # sequence must reproduce the model's own (uncropped) self_attn output
    # bit-for-bit -- the causal mask already zeros every j > i the crop
    # would otherwise include, so extending the crop window to T changes
    # nothing the real attention didn't already compute.
    for name, model in MODELS.items():
        ids4a = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
        captured = {}

        def cap_hook(mod, inp, out, captured=captured):
            captured["attn_out"] = out[0].detach()

        h = model.model.layers[0].self_attn.register_forward_hook(cap_hook)
        with torch.no_grad():
            model(input_ids=ids4a, attention_mask=torch.ones_like(ids4a), output_attentions=True)
        h.remove()
        real_out = captured["attn_out"]

        rows4a = [("3 4 5 6 7 8 9", 6)]  # prompt_len == seq_len - 1 < T: keep >=1 response token

        class _SpaceTok:
            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                ids = [int(c) for c in text.split()]
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor([ids], dtype=torch.long)}
                return {"input_ids": ids}

        # full-crop: prompt_len == T (no response rows averaged) isn't
        # expressible through cropped_attn_response_avg's own guard (it
        # needs >=1 response token) -- so this oracle calls the module's
        # inner mechanics directly on prompt_len == T via the SAME hook
        # pattern cropped_attn_response_avg uses, to compare PER-POSITION
        # (not response-averaged) cropped output against the real one.
        from steergym.gcad.rope import repeat_kv as _repeat_kv
        nh, nkv = model.config.num_attention_heads, model.config.num_key_value_heads
        hd = layer_dims(model)[3]
        captured_v, cropped = {}, {}

        def v_hook(mod, inp, out, captured_v=captured_v):
            captured_v[0] = out.detach()

        def attn_hook_full(mod, inp, out, cropped=cropped, captured_v=captured_v,
                           nh=nh, nkv=nkv, hd=hd):
            attn_weights = out[1]
            v_raw = captured_v.pop(0)
            bsz, seq_len_v = v_raw.shape[0], v_raw.shape[1]
            v = v_raw.view(bsz, seq_len_v, nkv, hd).transpose(1, 2)
            v = _repeat_kv(v, nh // nkv)
            crop_out = (attn_weights @ v).transpose(1, 2).reshape(bsz, -1, nh * hd)
            cropped[0] = mod.o_proj(crop_out).detach()

        hv = model.model.layers[0].self_attn.v_proj.register_forward_hook(v_hook)
        ha = model.model.layers[0].self_attn.register_forward_hook(attn_hook_full)
        with torch.no_grad():
            model(input_ids=ids4a, attention_mask=torch.ones_like(ids4a), output_attentions=True)
        hv.remove()
        ha.remove()
        check(f"{name}: Eq. 6/7 full-crop reproduces real self_attn output exactly (4a)",
              torch.equal(cropped[0], real_out))

    # ---- 4b. extraction shape/value checks + de-tautologized cross-check
    for name, model in MODELS.items():
        class _SpaceTok2:
            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                ids = [int(c) for c in text.split()]
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor([ids], dtype=torch.long)}
                return {"input_ids": ids}

        tok4b = _SpaceTok2()
        rows4b = [("3 4 5 6 7 8 9", 4, 2), ("10 11 12 13 14 15", 2, 2)]
        layer_list4b = [0, 1]
        avg_out = cropped_attn_response_avg(model, tok4b, rows4b, layer_list4b)
        check(f"{name}: cropped_attn_response_avg shapes [N, d] per layer (4b)",
              all(tuple(avg_out[l].shape) == (len(rows4b), model.config.hidden_size)
                  for l in layer_list4b))

        kpos, avgd = k_pos_and_avg_demand(model, tok4b, rows4b, layer_list4b)
        nh4b, nkv4b = model.config.num_attention_heads, model.config.num_key_value_heads
        hd4b = layer_dims(model)[3]
        check(f"{name}: k_pos shape [Lc, num_kv_heads, head_dim] (4b)",
              tuple(kpos.shape) == (len(layer_list4b), nkv4b, hd4b))
        check(f"{name}: avg_demand shape [Lc] (4b)", tuple(avgd.shape) == (len(layer_list4b),))
        check(f"{name}: k_pos/avg_demand are float32 (4b)",
              kpos.dtype == torch.float32 and avgd.dtype == torch.float32)

        # de-tautologized: recompute row 0's layer-0 response average by hand
        # from raw captured tensors, independent of cropped_attn_response_avg's
        # own internals.
        text0, plen0, crop0 = rows4b[0]
        ids0 = torch.tensor([tok4b(text0)["input_ids"]], dtype=torch.long)
        cap_v0, cap_w0 = {}, {}

        def vh(mod, inp, out, cap_v0=cap_v0):
            cap_v0["v"] = out.detach()

        def ah(mod, inp, out, cap_w0=cap_w0):
            cap_w0["w"] = out[1].detach()

        hv0 = model.model.layers[0].self_attn.v_proj.register_forward_hook(vh)
        ha0 = model.model.layers[0].self_attn.register_forward_hook(ah)
        with torch.no_grad():
            model(input_ids=ids0, attention_mask=torch.ones_like(ids0), output_attentions=True)
        hv0.remove()
        ha0.remove()
        v0 = cap_v0["v"].view(1, ids0.shape[1], nkv4b, hd4b).transpose(1, 2)
        from steergym.gcad.rope import repeat_kv as _rkv
        v0 = _rkv(v0, nh4b // nkv4b)
        crop_w0 = cap_w0["w"][:, :, :, :crop0]
        crop_v0 = v0[:, :, :crop0, :]
        crop_out0 = (crop_w0 @ crop_v0).transpose(1, 2).reshape(1, -1, nh4b * hd4b)
        expected0 = model.model.layers[0].self_attn.o_proj(crop_out0)[:, plen0:, :].mean(dim=1)
        check(f"{name}: cropped_attn_response_avg row 0/layer 0 matches an independent "
              "hand computation (4b)",
              torch.allclose(avg_out[0][0:1], expected0.float(), atol=1e-6))

    # ---- 4c. GCADContext: degenerate limit + positions + diagnostics
    for name, model in MODELS.items():
        Lc = len(model.model.layers)
        d = model.config.hidden_size
        nh, nkv = model.config.num_attention_heads, model.config.num_key_value_heads
        hd = layer_dims(model)[3]
        g4c = torch.Generator().manual_seed(21)
        v4c = torch.randn(Lc, d, generator=g4c) * 0.2
        kpos4c = torch.randn(Lc, nkv, hd, generator=g4c) * 0.2
        avgd4c = torch.rand(Lc, generator=g4c) * 0.1
        layers4c = list(model.model.layers)
        ids4c = torch.tensor([[3, 4, 5, 6, 7]], dtype=torch.long)

        with torch.no_grad():
            base4c = model(input_ids=ids4c, attention_mask=torch.ones_like(ids4c)).logits

        ctx_zero = GCADContext(layers4c, list(range(Lc)), v4c.clone(), kpos4c.clone(),
                               avgd4c.clone(), num_heads=nh, num_kv_heads=nkv, head_dim=hd,
                               base_coef=0.0, scale=1.5, positions="generated")
        with ctx_zero, torch.no_grad():
            zero4c = model(input_ids=ids4c, attention_mask=torch.ones_like(ids4c)).logits
        check(f"{name}: base_coef=0 (strength=0) is an EXACT identity (4c)",
              torch.equal(base4c, zero4c))

        ctx_nz = GCADContext(layers4c, list(range(Lc)), v4c.clone(), kpos4c.clone(),
                             avgd4c.clone(), num_heads=nh, num_kv_heads=nkv, head_dim=hd,
                             base_coef=3.0, scale=1.5, positions="generated")
        with ctx_nz, torch.no_grad():
            gen4c = model(input_ids=ids4c, attention_mask=torch.ones_like(ids4c)).logits
        check(f"{name}: nonzero base_coef changes the output (4c)",
              not torch.equal(base4c, gen4c))
        check(f"{name}: positions='generated' diagnostics counts one row per layer (4c)",
              ctx_nz.diagnostics()["steered_rows"] == Lc)

        ctx_all = GCADContext(layers4c, list(range(Lc)), v4c.clone(), kpos4c.clone(),
                              avgd4c.clone(), num_heads=nh, num_kv_heads=nkv, head_dim=hd,
                              base_coef=3.0, scale=1.5, positions="all")
        with ctx_all, torch.no_grad():
            all4c = model(input_ids=ids4c, attention_mask=torch.ones_like(ids4c)).logits
        check(f"{name}: positions='all' diagnostics counts every position per layer (4c)",
              ctx_all.diagnostics()["steered_rows"] == Lc * ids4c.shape[1])
        check(f"{name}: positions='all' != positions='generated' output (4c)",
              not torch.equal(all4c, gen4c))

        # re-entry guard
        reentry_ok = False
        try:
            with ctx_nz:
                with ctx_nz:
                    pass
        except RuntimeError:
            reentry_ok = True
        check(f"{name}: GCADContext re-entry raises RuntimeError (4c)", reentry_ok)

    # ---- 4d. KV-cache generate equivalence: batched (left-padded) vs alone
    for name, model in MODELS.items():
        Lc = len(model.model.layers)
        d = model.config.hidden_size
        nh, nkv = model.config.num_attention_heads, model.config.num_key_value_heads
        hd = layer_dims(model)[3]
        g4d = torch.Generator().manual_seed(22)
        v4d = torch.randn(Lc, d, generator=g4d) * 0.2
        kpos4d = torch.randn(Lc, nkv, hd, generator=g4d) * 0.2
        avgd4d = torch.rand(Lc, generator=g4d) * 0.1
        layers4d = list(model.model.layers)

        long_ids = torch.randint(3, HF_VOCAB, (1, 7), generator=g4d)
        short_ids = torch.randint(3, HF_VOCAB, (1, 4), generator=g4d)
        pad4d = torch.zeros(1, 3, dtype=torch.long)
        batch4d = torch.cat([torch.cat([pad4d, short_ids], 1), long_ids], 0)
        am4d = torch.cat([torch.cat([torch.zeros(1, 3, dtype=torch.long),
                                     torch.ones(1, 4, dtype=torch.long)], 1),
                         torch.ones(1, 7, dtype=torch.long)], 0)

        def _gen(ids, am):
            ctx = GCADContext(layers4d, list(range(Lc)), v4d.clone(), kpos4d.clone(),
                              avgd4d.clone(), num_heads=nh, num_kv_heads=nkv, head_dim=hd,
                              base_coef=2.0, scale=1.5, positions="generated")
            with ctx, torch.no_grad():
                return model.generate(input_ids=ids, attention_mask=am, max_new_tokens=4,
                                      min_new_tokens=4, do_sample=False, pad_token_id=0)

        out_batch = _gen(batch4d, am4d)
        out_alone = _gen(short_ids, torch.ones_like(short_ids))
        check(f"{name}: KV-cache generate, left-padded batch row0 == same prompt alone (4d)",
              torch.equal(out_batch[0, 7:], out_alone[0, 4:]))

    # ---- 4e. attn_implementation requirement
    torch.manual_seed(11)
    cfg_sdpa = _Qwen2Config(vocab_size=HF_VOCAB, hidden_size=HF_D, intermediate_size=2 * HF_D,
                            num_hidden_layers=HF_LAYERS, num_attention_heads=HF_HEADS,
                            num_key_value_heads=HF_KV_HEADS, max_position_embeddings=64,
                            pad_token_id=0, attn_implementation="sdpa")
    model_sdpa = _Qwen2ForCausalLM(cfg_sdpa).eval()

    class _SpaceTok3:
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = [int(c) for c in text.split()]
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
            return {"input_ids": ids}

    raised = False
    try:
        cropped_attn_response_avg(model_sdpa, _SpaceTok3(), [("3 4 5 6 7", 2, 2)], [0])
    except RuntimeError:
        raised = True
    check("sdpa attn_implementation raises a clear RuntimeError (needs eager) (4e)", raised)

    # ---- 4f. fit() end-to-end + determinism + save/load round trip
    import polars as pl

    BOS, EOS, PAD = 1, 2, 0

    class _BatchDict(dict):
        def to(self, device):
            return self

    class CharChatTok:
        pad_token_id = PAD
        eos_token_id = EOS
        bos_token = "<bos>"
        padding_side = "right"

        def _t(self, c):
            return 3 + (ord(c) % 250)

        def decode(self, ids):
            return "".join(chr(3 + (int(i) - 3) % 250) for i in ids if int(i) != BOS)

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            s = "<<" + "".join(m["content"] for m in messages) + "end"
            if add_generation_prompt:
                s += ">>"
            if not tokenize:
                return self.bos_token + s
            return [BOS] + [self._t(c) for c in s]

        def __call__(self, text, return_tensors=None, add_special_tokens=True,
                    max_length=None, truncation=None):
            ids = [self._t(c) for c in text]
            if add_special_tokens:
                ids = [BOS] + ids
            if truncation and max_length:
                ids = ids[:max_length]
            if return_tensors == "pt":
                return _BatchDict(input_ids=torch.tensor([ids], dtype=torch.long))
            return _BatchDict(input_ids=ids)

    def mk_row(cid, prompt, response, oc, cat, genre="text"):
        return {"concept_id": cid, "prompt": prompt, "response": response,
                "meta": {"output_concept": oc, "category": cat, "concept_genre": genre}}

    rows4f = []
    for i in range(3):
        rows4f.append(mk_row(0, f"tell about A{i}", "aaaa" + str(i), "conceptA", "positive"))
    for i in range(3):
        rows4f.append(mk_row(-1, f"neutral {i}", "nnnn" + str(i), "", "negative"))
    samples4f = pl.DataFrame(rows4f)
    concepts4f = [Concept(id=0, name="conceptA", description="")]

    for name, model in MODELS.items():
        tok4f = CharChatTok()
        cfg4f = GCADConfig(subject=SUBJECT, layers=(0, 1))
        m1 = GCADModel(cfg4f).bind(model, tok4f)
        m1.fit(concepts4f, samples4f)

        c = m1.concepts[0]
        Lc4f = len(cfg4f.layers)
        d4f = model.config.hidden_size
        nkv4f = model.config.num_key_value_heads
        hd4f = layer_dims(model)[3]
        check(f"{name}: fit() v shape [Lc, d] (4f)", tuple(c["v"].shape) == (Lc4f, d4f))
        check(f"{name}: fit() k_pos shape [Lc, num_kv_heads, head_dim] (4f)",
              tuple(c["k_pos"].shape) == (Lc4f, nkv4f, hd4f))
        check(f"{name}: fit() avg_demand shape [Lc] (4f)",
              tuple(c["avg_demand"].shape) == (Lc4f,))

        m2 = GCADModel(cfg4f).bind(model, tok4f)
        m2.fit(concepts4f, samples4f)
        check(f"{name}: fit() is deterministic (no RNG) across two runs (4f)",
              torch.equal(m1.concepts[0]["v"], m2.concepts[0]["v"])
              and torch.equal(m1.concepts[0]["k_pos"], m2.concepts[0]["k_pos"])
              and torch.equal(m1.concepts[0]["avg_demand"], m2.concepts[0]["avg_demand"]))

        import tempfile
        path4f = tempfile.mktemp()
        m1.save(path4f)
        loaded4f = GCADModel.load(path4f)
        os.remove(path4f)
        check(f"{name}: save/load round trip preserves v/k_pos/avg_demand (4f)",
              torch.equal(loaded4f.concepts[0]["v"], c["v"])
              and torch.equal(loaded4f.concepts[0]["k_pos"], c["k_pos"])
              and torch.equal(loaded4f.concepts[0]["avg_demand"], c["avg_demand"]))
        check(f"{name}: save/load preserves layer_list (4f)", loaded4f.layer_list == m1.layer_list)

        loaded4f.bind(model, tok4f)
        context_ok = True
        try:
            with loaded4f.context(concepts4f[0], strength=1.0) as ctx4f, torch.no_grad():
                model(input_ids=torch.tensor([[3, 4, 5, 6, 7]]),
                     attention_mask=torch.ones(1, 5, dtype=torch.long))
        except Exception:
            context_ok = False
        check(f"{name}: a loaded model's context() runs generation without error (4f)",
              context_ok)

        # corrupt-artifact load rejection
        bad_raises = False
        bad_path = tempfile.mktemp()
        import torch as _torch
        d_bad = _torch.load(path4f, map_location="cpu", weights_only=False) if False else None
        m1.save(bad_path)
        state = torch.load(bad_path, map_location="cpu", weights_only=False)
        del state["concepts"][0]["avg_demand"]
        torch.save(state, bad_path)
        try:
            GCADModel.load(bad_path)
        except ValueError:
            bad_raises = True
        os.remove(bad_path)
        check(f"{name}: load() rejects an artifact with a missing key (4f)", bad_raises)


# ================================================================== 5. GCAD_DIR-gated upstream source parity
GCAD_DIR = os.environ.get("GCAD_DIR")
if not GCAD_DIR:
    print("SKIP upstream source-parity leg: GCAD_DIR not set")
elif _real_transformers is None:
    print("SKIP upstream source-parity leg: transformers unavailable")
else:
    sys.path.insert(0, os.path.join(GCAD_DIR, "core"))
    sys.path.insert(0, os.path.join(GCAD_DIR, "scripts"))
    try:
        import generate_vec as upstream_genvec
        import generate_vec_prompt_attn_k as upstream_kpos
        import gated_steerer as upstream_gated
    except Exception as exc:  # pandas/tqdm/etc. missing
        print(f"SKIP upstream source-parity leg: import failed ({exc!r})")
        upstream_genvec = None

    if upstream_genvec is not None:
        from steergym.gcad.extract import cropped_attn_response_avg, k_pos_and_avg_demand

        torch.manual_seed(30)
        up_cfg = _Qwen2Config(vocab_size=300, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                              max_position_embeddings=64, attn_implementation="eager",
                              pad_token_id=0)
        up_model = _Qwen2ForCausalLM(up_cfg).eval()
        for p in up_model.parameters():
            p.requires_grad_(False)

        UP_BOS = 1

        class _UpBatchDict(dict):
            def to(self, device):
                return self

        class UpstreamTok:
            def _t(self, c):
                return 3 + (ord(c) % 250)

            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                ids = [self._t(c) for c in text]
                if add_special_tokens:
                    ids = [UP_BOS] + ids
                if return_tensors == "pt":
                    return _UpBatchDict(input_ids=torch.tensor([ids], dtype=torch.long))
                return _UpBatchDict(input_ids=ids)

            def encode(self, text, add_special_tokens=True):
                ids = [self._t(c) for c in text]
                if add_special_tokens:
                    ids = [UP_BOS] + ids
                return ids

        up_tok = UpstreamTok()

        # Non-empty user turn: our crop (system block) must sit strictly
        # inside the prompt, exactly at upstream's "<|im_end|>\n" boundary.
        SYS_MARK = "<|im_end|>\n"
        sys5 = "sysblock" + SYS_MARK
        prompt5 = sys5 + "user asks"
        response5 = "the model response continues here"
        text5 = prompt5 + response5
        prompt_len5 = len(up_tok.encode(prompt5, add_special_tokens=False))
        crop_len5 = len(up_tok.encode(sys5, add_special_tokens=False))
        check("upstream parity: crop window strictly inside the prompt (5)", crop_len5 < prompt_len5)
        rows5 = [(text5, prompt_len5, crop_len5)]

        _, _, up_resp_avg = upstream_genvec.get_prompt_attn_delta_p_and_r(
            up_model, up_tok, [prompt5], [response5], layer_list=[0, 1])
        our_avg = cropped_attn_response_avg(up_model, up_tok, rows5, [0, 1])
        delta_7 = max((up_resp_avg[l] - our_avg[l]).abs().max().item() for l in [0, 1])
        check("upstream parity: Eq. 7/8 cropped-attn-delta extraction, max abs diff == 0 (5)",
              delta_7 == 0.0)

        up_kpos, up_avgd = upstream_kpos.compute_k_pos_and_avg_demand(
            up_model, up_tok, [prompt5], [response5], layer_list=[0, 1])
        our_kpos, our_avgd = k_pos_and_avg_demand(up_model, up_tok, rows5, [0, 1])
        check("upstream parity: Eq. 9 k_pos extraction, max abs diff == 0 (5)",
              (up_kpos - our_kpos).abs().max().item() == 0.0)
        check("upstream parity: Eq. 9 avg_demand extraction, max abs diff == 0 (5)",
              (up_avgd - our_avgd).abs().max().item() == 0.0)

        Lc5 = 2
        d5 = up_cfg.hidden_size
        g5 = torch.Generator().manual_seed(31)
        v_dict = {l: torch.randn(d5, generator=g5) * 0.1 for l in range(Lc5)}
        k_pos_dict = {l: torch.randn(up_cfg.num_key_value_heads,
                                     d5 // up_cfg.num_attention_heads, generator=g5) * 0.1
                     for l in range(Lc5)}
        avg_demand_dict = {l: 0.05 * (l + 1) for l in range(Lc5)}

        ids5 = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
        with upstream_gated.MultiLayerGatedSteerer(
                up_model, v_dict, k_pos_dict, avg_demand_dict, layer_indices=[0, 1],
                base_coef=2.0, scale=1.5):
            with torch.no_grad():
                up_out = up_model(input_ids=ids5, attention_mask=torch.ones_like(ids5)).logits

        v_t5 = torch.stack([v_dict[l] for l in range(Lc5)], dim=0)
        kpos_t5 = torch.stack([k_pos_dict[l] for l in range(Lc5)], dim=0)
        avgd_t5 = torch.tensor([avg_demand_dict[l] for l in range(Lc5)])
        layers5 = [up_model.model.layers[l] for l in range(Lc5)]
        ctx5 = GCADContext(layers5, list(range(Lc5)), v_t5, kpos_t5, avgd_t5,
                           num_heads=up_cfg.num_attention_heads,
                           num_kv_heads=up_cfg.num_key_value_heads,
                           head_dim=d5 // up_cfg.num_attention_heads,
                           base_coef=2.0, scale=1.5, positions="all")
        with ctx5, torch.no_grad():
            our_out5 = up_model(input_ids=ids5, attention_mask=torch.ones_like(ids5)).logits
        check("upstream parity: Eq. 10/11 injection (MultiLayerGatedSteerer vs GCADContext), "
              "max abs diff == 0 (5)", (up_out - our_out5).abs().max().item() == 0.0)


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
