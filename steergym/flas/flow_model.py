"""FLAS FlowBlock architecture (arXiv 2605.05892; upstream github
flas-ai/FLAS, src/flas/model.py). Port scope: inference only -- the
training-time constructors (`build_flow_model`, `ConceptEncoder.__init__`'s
from-scratch load, `build_chat_prompt`) are dropped; `ConceptEncoder` is
always built by sharing modules with an already-loaded subject model
(`from_base_model`). See steergym/flas/README.md for the velocity-field
equation, the registry, and every upstream delta.

Module level carries torch/nn (nn.Module subclassing needs them bound at
class-definition time, the acttransport/maps.py precedent); every
`transformers` import stays lazy inside the function that needs it, so a
bare `import steergym` never touches it.
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _apply_rope_single(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class _RopeLayerTypeWrapper(nn.Module):
    """Adapts a layer-type-keyed RoPE (Gemma3RotaryEmbedding) to the plain
    rope(x, position_ids) -> (cos, sin) call FlowCrossAttention expects."""

    def __init__(self, rope, layer_type):
        super().__init__()
        self.rope = rope
        self.layer_type = layer_type

    def forward(self, x, position_ids):
        return self.rope(x, position_ids, layer_type=self.layer_type)


@dataclass
class ArchSpec:
    """Per-family modules + config-derived behaviors, resolved once per
    model by `get_arch_spec(config)` keyed on `config.model_type`."""
    rms_norm_cls: Type[nn.Module]
    rotary_emb_cls: Type[nn.Module]
    attention_cls: Type[nn.Module]
    embed_scale: bool
    needs_layer_types: bool
    self_attn_layer_idx_fn: Callable[[int], int]
    norm_init_map: Dict[str, Optional[str]]
    rope_layer_type: Optional[str] = None


def _gemma2_spec():
    from transformers.models.gemma2.modeling_gemma2 import (
        Gemma2Attention, Gemma2RMSNorm, Gemma2RotaryEmbedding)
    return ArchSpec(
        rms_norm_cls=Gemma2RMSNorm, rotary_emb_cls=Gemma2RotaryEmbedding,
        attention_cls=Gemma2Attention, embed_scale=True, needs_layer_types=False,
        self_attn_layer_idx_fn=lambda layer_idx: layer_idx % 2,  # even=sliding, odd=global
        norm_init_map={"pre_sa_norm": "input_layernorm", "post_sa_norm": "post_attention_layernorm",
                      "pre_mlp_norm": "pre_feedforward_layernorm", "post_mlp_norm": "post_feedforward_layernorm"})


def _qwen3_spec():
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention, Qwen3RMSNorm, Qwen3RotaryEmbedding)
    return ArchSpec(
        rms_norm_cls=Qwen3RMSNorm, rotary_emb_cls=Qwen3RotaryEmbedding,
        attention_cls=Qwen3Attention, embed_scale=False, needs_layer_types=True,
        self_attn_layer_idx_fn=lambda layer_idx: 0,
        norm_init_map={"pre_sa_norm": "input_layernorm", "post_sa_norm": None,
                      "pre_mlp_norm": "post_attention_layernorm", "post_mlp_norm": None})


def _llama_spec():
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding)
    return ArchSpec(
        rms_norm_cls=LlamaRMSNorm, rotary_emb_cls=LlamaRotaryEmbedding,
        attention_cls=LlamaAttention, embed_scale=False, needs_layer_types=True,
        self_attn_layer_idx_fn=lambda layer_idx: 0,
        norm_init_map={"pre_sa_norm": "input_layernorm", "post_sa_norm": None,
                      "pre_mlp_norm": "post_attention_layernorm", "post_mlp_norm": None})


def _gemma3_spec():
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3Attention, Gemma3RMSNorm, Gemma3RotaryEmbedding)
    return ArchSpec(
        rms_norm_cls=Gemma3RMSNorm, rotary_emb_cls=Gemma3RotaryEmbedding,
        attention_cls=Gemma3Attention, embed_scale=True, needs_layer_types=True,
        self_attn_layer_idx_fn=lambda layer_idx: 0,
        norm_init_map={"pre_sa_norm": "input_layernorm", "post_sa_norm": "post_attention_layernorm",
                      "pre_mlp_norm": "pre_feedforward_layernorm", "post_mlp_norm": "post_feedforward_layernorm"},
        rope_layer_type="full_attention")


_ARCH_REGISTRY: Dict[str, Callable[[], ArchSpec]] = {
    "gemma2": _gemma2_spec, "qwen3": _qwen3_spec, "llama": _llama_spec,
    "gemma3": _gemma3_spec, "gemma3_text": _gemma3_spec,
}


def get_arch_spec(config) -> ArchSpec:
    mt = getattr(config, "model_type", None)
    if mt not in _ARCH_REGISTRY:
        raise ValueError(f"unsupported model_type {mt!r}; supported: {sorted(_ARCH_REGISTRY)}")
    return _ARCH_REGISTRY[mt]()


def get_text_config(config):
    # multimodal wrappers (gemma3-4b) nest the decoder config under .text_config
    return getattr(config, "text_config", None) or config


def get_text_decoder(model):
    # multimodal wrappers (Gemma3ForConditionalGeneration) nest the decoder
    # under model.model.language_model; text-only models expose it at model.model
    base = model.model
    if hasattr(base, "language_model"):
        return base.language_model
    return base


class FlowCrossAttention(nn.Module):
    """GQA cross-attention: activation queries (Q=h) attend to the encoded
    concept (KV). Family-agnostic -- rms_norm_cls/rotary_emb are injected so
    this module never imports a family's modeling classes."""

    def __init__(self, *, hidden_size, num_heads, num_kv_heads, head_dim,
                 rms_norm_eps, rotary_emb, rms_norm_cls,
                 attn_bias=False, softcap=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.softcap = softcap

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=attn_bias)

        self.q_norm = rms_norm_cls(self.head_dim, eps=rms_norm_eps)
        self.k_norm = rms_norm_cls(self.head_dim, eps=rms_norm_eps)
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None,
                q_pos_offset=0, q_position_ids=None, kv_position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        kv_len = encoder_hidden_states.size(1)

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).view(bsz, kv_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).view(bsz, kv_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)

        q_pos = q_position_ids if q_position_ids is not None else (
            torch.arange(q_len, device=q.device).unsqueeze(0) + q_pos_offset)
        kv_pos = kv_position_ids if kv_position_ids is not None else (
            torch.arange(kv_len, device=k.device).unsqueeze(0))
        q_cos, q_sin = self.rotary_emb(q, q_pos)
        k_cos, k_sin = self.rotary_emb(k, kv_pos)
        q = _apply_rope_single(q, q_cos, q_sin)
        k = _apply_rope_single(k, k_cos, k_sin)

        k_exp = repeat_kv(k, self.num_kv_groups)
        v_exp = repeat_kv(v, self.num_kv_groups)
        attn_weights = torch.matmul(q, k_exp.transpose(2, 3)) * self.scaling

        if self.softcap is not None:
            attn_weights = torch.tanh(attn_weights / self.softcap) * self.softcap
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask[:, None, None, :]
            attn_weights = attn_weights.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_weights, v_exp)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out)


def sinusoidal_time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=128):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t):
        emb = sinusoidal_time_embedding(t, self.freq_dim)
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


class FlowBlock(nn.Module):
    """TimeEmbed -> CrossAttn(Q=h, KV=concept) -> Causal SelfAttn(over h's
    own sequence) -> gated MLP, each wrapped in a residual + learnable
    per-channel gate. The self-attn KV cache is OURS (steergym.flas's own
    DynamicCache per Euler step, see README) -- cross-attn's concept KV is
    recomputed fresh every call (small, no cache needed)."""

    def __init__(self, config, spec: ArchSpec, rotary_emb, init_gate=0.1,
                 layer_idx=0, disable_cross_attn=False, disable_self_attn=False, disable_mlp=False):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        rms_norm_eps = config.rms_norm_eps
        RMSNorm = spec.rms_norm_cls
        self.layer_idx = layer_idx
        self.disable_cross_attn = disable_cross_attn
        self.disable_self_attn = disable_self_attn
        self.disable_mlp = disable_mlp

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        attn_bias = getattr(config, "attention_bias", False)
        softcap = getattr(config, "attn_logit_softcapping", None)

        self.pre_cross_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_attn = FlowCrossAttention(
            hidden_size=hidden_size, num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads, head_dim=head_dim,
            rms_norm_eps=rms_norm_eps, rotary_emb=rotary_emb, rms_norm_cls=RMSNorm,
            attn_bias=attn_bias, softcap=softcap)
        self.post_cross_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        if not disable_self_attn:
            sa_config = type(config).from_dict(config.to_dict())
            sa_config._attn_implementation = "eager"
            if spec.needs_layer_types:
                sa_config.layer_types = ["full_attention"] * sa_config.num_hidden_layers
            self.pre_sa_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.post_sa_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            sa_layer_idx = spec.self_attn_layer_idx_fn(layer_idx)
            self.self_attn = spec.attention_cls(sa_config, layer_idx=sa_layer_idx)
            self.self_attn_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        self.pre_mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.post_mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        act_name = getattr(config, "hidden_act", None) or getattr(config, "hidden_activation", None)
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[act_name]
        self.mlp_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

    def forward(self, h, concept_hidden, concept_mask=None, t_emb=None, self_attn_cache=None,
                position_embeddings=None, padding_mask=None, use_cache=False,
                q_pos_offset=0, activation_position_ids=None):
        if t_emb is not None:
            h = h + t_emb[:, None, :]

        if not self.disable_cross_attn:
            h_normed = self.pre_cross_norm(h)
            ca_delta = self.cross_attn(h_normed, concept_hidden, concept_mask,
                                      q_pos_offset=q_pos_offset, q_position_ids=activation_position_ids)
            ca_delta = self.post_cross_norm(ca_delta)
            h = h + self.cross_gate * ca_delta

        if self.disable_self_attn:
            new_cache = self_attn_cache if use_cache else None
        else:
            h_normed_sa = self.pre_sa_norm(h)
            q_len = h.size(1)
            past_len = self_attn_cache.get_seq_length() if self_attn_cache is not None else 0
            kv_len = past_len + q_len

            MASK_VAL = -1e4
            row_idx = torch.arange(q_len, device=h.device).unsqueeze(1)
            col_idx = torch.arange(kv_len, device=h.device).unsqueeze(0)
            causal_mask = torch.where(col_idx <= row_idx + past_len,
                                      torch.zeros(1, device=h.device, dtype=h.dtype),
                                      torch.full((1,), MASK_VAL, device=h.device, dtype=h.dtype))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            if padding_mask is not None:
                pm = padding_mask[:, :kv_len]
                pad_4d = (1.0 - pm[:, None, None, :].to(h.dtype)) * MASK_VAL
                causal_mask = causal_mask + pad_4d

            sa_out = self.self_attn(h_normed_sa, position_embeddings=position_embeddings,
                                    attention_mask=causal_mask, past_key_values=self_attn_cache,
                                    cache_position=None)
            sa_delta = sa_out[0]
            new_cache = self_attn_cache if use_cache else None
            sa_delta = self.post_sa_norm(sa_delta)
            h = h + self.self_attn_gate * sa_delta

        if not self.disable_mlp:
            x = self.pre_mlp_norm(h)
            x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            x = self.post_mlp_norm(x)
            h = h + self.mlp_gate * x

        return h, new_cache


class FlowFunction(nn.Module):
    """Velocity field v_theta(h, t, concept) = blocks(h, t, c) - h_input."""

    def __init__(self, config, num_blocks=1, time_conditioned=True, layer_idx=0,
                 disable_cross_attn=False, disable_self_attn=False, disable_mlp=False):
        super().__init__()
        self.config = config
        self.spec = get_arch_spec(config)
        self.hidden_size = config.hidden_size
        self.num_blocks = num_blocks
        self.time_conditioned = time_conditioned

        if time_conditioned:
            self.time_embed = TimeEmbedder(config.hidden_size)

        base_rope = self.spec.rotary_emb_cls(config=config)
        self.rotary_emb = (_RopeLayerTypeWrapper(base_rope, self.spec.rope_layer_type)
                           if self.spec.rope_layer_type is not None else base_rope)

        self.blocks = nn.ModuleList(
            [FlowBlock(config, self.spec, self.rotary_emb, layer_idx=layer_idx,
                      disable_cross_attn=disable_cross_attn,
                      disable_self_attn=disable_self_attn, disable_mlp=disable_mlp)
             for _ in range(num_blocks)])

    def forward(self, h, concept_hidden, concept_mask=None, t=None, self_attn_caches=None,
                use_cache=False, padding_mask=None, past_len=0, position_ids=None):
        h_input = h
        t_emb = self.time_embed(t) if self.time_conditioned and t is not None else None

        seq_len = h.size(1)
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=h.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(h, position_ids)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if self_attn_caches is not None:
                sc = self_attn_caches[i]
            elif use_cache:
                from transformers.cache_utils import DynamicCache
                sc = DynamicCache()
            else:
                sc = None
            h, kv_cache = block(h, concept_hidden, concept_mask, t_emb=t_emb, self_attn_cache=sc,
                                position_embeddings=position_embeddings, padding_mask=padding_mask,
                                use_cache=use_cache, q_pos_offset=past_len, activation_position_ids=position_ids)
            if use_cache:
                new_caches.append(kv_cache)

        return h - h_input, new_caches


class ConceptEncoder(nn.Module):
    """Frozen N-layer base-model encoder for the steering concept text,
    sharing modules with an already-loaded subject LM (no second copy of
    the embedding table / first N decoder layers). Training-time variant
    that loads its own full copy (upstream's regular `__init__`) is out of
    scope -- this port is inference-only."""

    def __init__(self, embed_tokens, layers, norm, rotary_emb, hidden_size, spec, layer_types):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = layers
        self.norm = norm
        self.rotary_emb = rotary_emb
        self.hidden_size = hidden_size
        self.spec = spec
        self._layer_types = layer_types
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def from_base_model(cls, base_model, num_layers=2):
        cfg = get_text_config(base_model.config)
        spec = get_arch_spec(cfg)
        base = get_text_decoder(base_model)
        layer_types = list(getattr(cfg, "layer_types", None) or [])[:num_layers]
        return cls(base.embed_tokens, nn.ModuleList(list(base.layers[:num_layers])),
                  base.norm, base.rotary_emb, cfg.hidden_size, spec, layer_types)

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)
        if self.spec.embed_scale and not hasattr(self.embed_tokens, "embed_scale"):
            h = h * (self.hidden_size ** 0.5)

        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)

        MIN_VAL = torch.finfo(h.dtype).min
        causal = torch.triu(torch.full((seq_len, seq_len), MIN_VAL, device=h.device, dtype=h.dtype),
                            diagonal=1).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * MIN_VAL
            mask_4d = (causal + pad_mask).clamp(min=MIN_VAL)
        else:
            mask_4d = causal.expand(bsz, -1, -1, -1)

        if self.spec.rope_layer_type is not None and self._layer_types:
            pe_by_type = {lt: self.rotary_emb(h, position_ids, layer_type=lt)
                         for lt in set(self._layer_types)}
            for layer, lt in zip(self.layers, self._layer_types):
                out = layer(h, attention_mask=mask_4d, position_ids=position_ids,
                           position_embeddings=pe_by_type[lt])
                h = out[0] if isinstance(out, tuple) else out
        else:
            position_embeddings = self.rotary_emb(h, position_ids)
            for layer in self.layers:
                out = layer(h, attention_mask=mask_4d, position_ids=position_ids,
                           position_embeddings=position_embeddings)
                h = out[0] if isinstance(out, tuple) else out

        return self.norm(h)
