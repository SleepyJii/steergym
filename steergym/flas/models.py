"""CachedFLASSteerModel: inference-only wrapper around the published FLAS
FlowBlock checkpoints (arXiv 2605.05892). "Cached" -- no fit/training path;
weights come from a frozen registry of HF repos, downloaded lazily at
bind/load time. See README.md for the registry provenance, the velocity
field, and the decoding-cache design."""
from dataclasses import dataclass

from ..concept import Concept
from ..steer import GeneratorRef, SteerConfig, SteerContext, SteerModel

# Registry of record: model_id/layer/num_blocks/n_steps/disable_* read from
# each repo's own config.json (2026-10-12); model_type/hidden_size read from
# each repo's weight.safetensors header (cross_attn.q_proj.weight shape) --
# see README.md "Registry" for the full derivation and citations.
FLAS_CHECKPOINTS = {
    "flas-ai/flas-gemma-2-2b-it": dict(
        model_id="google/gemma-2-2b-it", model_type="gemma2", hidden_size=2304,
        layer=20, num_blocks=1, n_steps=3, disable_self_attn=False, prompt_format="chat"),
    "flas-ai/flas-gemma-2-9b-it": dict(
        model_id="google/gemma-2-9b-it", model_type="gemma2", hidden_size=3584,
        layer=20, num_blocks=1, n_steps=3, disable_self_attn=False, prompt_format="chat"),
    "flas-ai/flas-llama-3.1-8b-instruct": dict(
        model_id="meta-llama/Llama-3.1-8B-Instruct", model_type="llama", hidden_size=4096,
        layer=20, num_blocks=1, n_steps=3, disable_self_attn=True, prompt_format="chat"),
    "flas-ai/flas-llama-3.1-8b": dict(
        model_id="meta-llama/Llama-3.1-8B", model_type="llama", hidden_size=4096,
        layer=20, num_blocks=1, n_steps=3, disable_self_attn=True, prompt_format="alpaca"),
    "flas-ai/flas-qwen3-8b": dict(
        model_id="Qwen/Qwen3-8B", model_type="qwen3", hidden_size=4096,
        layer=20, num_blocks=1, n_steps=3, disable_self_attn=True, prompt_format="chat"),
    "flas-ai/flas-qwen3-8b-base": dict(
        model_id="Qwen/Qwen3-8B-Base", model_type="qwen3", hidden_size=4096,
        layer=20, num_blocks=1, n_steps=3, disable_self_attn=True, prompt_format="alpaca"),
    "flas-ai/flas-gemma-3-4b-it": dict(
        model_id="google/gemma-3-4b-it", model_type="gemma3_text", hidden_size=2560,
        layer=20, num_blocks=1, n_steps=3, disable_self_attn=True, prompt_format="chat"),
    "flas-ai/flas-gemma-3-4b-pt": dict(
        model_id="google/gemma-3-4b-pt", model_type="gemma3_text", hidden_size=2560,
        layer=20, num_blocks=1, n_steps=3, disable_self_attn=True, prompt_format="alpaca"),
}


class FLASCheckpointError(ValueError):
    """hf_repo is not a published FLAS checkpoint, or the bound generator's
    model doesn't match the checkpoint's registry entry (model_type /
    hidden_size)."""


@dataclass
class CachedFLASConfig(SteerConfig):
    hf_repo: str = ""
    model_type: str = ""
    hidden_size: int = 0
    num_blocks: int = 1
    disable_cross_attn: bool = False
    disable_self_attn: bool = False
    disable_mlp: bool = False
    n_steps: int = 3
    concept_max_len: int = 64


def _padding_mask_from_attn_mask(attn_mask):
    # attn_mask is the subject model's OWN processed mask as handed to the
    # hooked decoder layer (eager attn_implementation: additive float,
    # shape [B, 1, q, kv] or a raw 2D [B, kv]); the last query row sees
    # every causally-eligible kv position, so where IT is masked is exactly
    # padding -- true for a full_attention steering layer with sequences
    # inside any sliding window (AxBench-length prompts, window >> prompt).
    if attn_mask is None:
        return None
    import torch

    last_row = attn_mask[:, 0, -1, :] if attn_mask.dim() == 4 else attn_mask
    if last_row.dtype == torch.bool:
        return last_row.float()
    return (last_row == 0).float()


class CachedFLASContext(SteerContext):
    """One forward hook on the steering layer's OUTPUT, running the
    n_steps-step forward-Euler integration h_{k+1} = h_k + dt*v(h_k, t_k, c)
    at EVERY position of the current forward call (prefill: every prompt
    token; decode: the one new token) -- FLAS has no positions="generated"
    vs "all" choice, unlike the row/transform families.

    The FlowBlock's own causal self-attention needs K/V from every PRIOR
    decode position at the SAME Euler step k; `_sa_caches[k]` is one
    DynamicCache per block per step, grown incrementally across calls
    (never recomputed from scratch) -- see README "Decoding-cache design".
    Prefill-vs-decode is inferred the way steergym.alqr.spid's SPIDContext
    does (q_len > 1, or no cache yet, or a batch-size change): a forward
    hook has no other signal, and a generate() call whose every prompt is
    a single token is indistinguishable from a decode step (same accepted
    blind spot, never reachable on real text)."""

    def __init__(self, module, flow_fn, concept_hidden, concept_mask, T: float, n_steps: int):
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        self.module = module
        self.flow_fn = flow_fn
        self.concept_hidden = concept_hidden  # [1, kv_len, d], flow_fn's dtype
        self.concept_mask = concept_mask      # [1, kv_len] float
        self.T = float(T)
        self.n_steps = int(n_steps)
        self._handle = None
        self._sa_caches = None
        self._past_len = 0
        self._prev_bsz = None
        self._dev_cache = None  # (device, concept_hidden, concept_mask)
        self._rows = 0
        self._delta_sum = 0.0
        self._delta_max = 0.0

    def _concept_on(self, device, dtype):
        cached = self._dev_cache
        if cached is None or cached[0] != device:
            cached = (device, self.concept_hidden.to(device=device, dtype=dtype),
                     self.concept_mask.to(device=device))
            self._dev_cache = cached
        return cached[1], cached[2]

    def _step(self, h_orig, attn_mask, position_ids):
        import torch

        flow_dtype = next(self.flow_fn.parameters()).dtype
        bsz, q_len, _ = h_orig.shape
        is_prefill = (self._sa_caches is None or q_len > 1 or bsz != self._prev_bsz)
        if is_prefill:
            self._sa_caches = [None] * self.n_steps
            self._past_len = 0

        if next(self.flow_fn.parameters()).device != h_orig.device:
            self.flow_fn.to(h_orig.device)
        concept_hidden, concept_mask = self._concept_on(h_orig.device, flow_dtype)
        padding_mask = _padding_mask_from_attn_mask(attn_mask)

        h = h_orig.to(flow_dtype)
        dt = self.T / self.n_steps
        for k in range(self.n_steps):
            t_k = torch.full((bsz,), dt * k, device=h.device, dtype=flow_dtype)
            v, kv_caches = self.flow_fn(
                h, concept_hidden.expand(bsz, -1, -1), concept_mask.expand(bsz, -1), t=t_k,
                self_attn_caches=None if is_prefill else self._sa_caches[k],
                padding_mask=padding_mask, use_cache=True,
                past_len=0 if is_prefill else self._past_len, position_ids=position_ids)
            self._sa_caches[k] = kv_caches
            h = h + dt * v

        self._past_len += q_len
        self._prev_bsz = bsz
        delta_norms = (h.detach() - h_orig.to(flow_dtype)).norm(dim=-1).reshape(-1)
        self._rows += len(delta_norms)
        self._delta_sum += float(delta_norms.sum())
        self._delta_max = max(self._delta_max, float(delta_norms.max()) if len(delta_norms) else 0.0)
        return h.to(h_orig.dtype)

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("context already entered (nested entry would leak a hook)")

        def hook(mod, args, kwargs, output):
            h_orig = output[0] if isinstance(output, tuple) else output
            h_out = self._step(h_orig, kwargs.get("attention_mask"), kwargs.get("position_ids"))
            return (h_out,) + tuple(output[1:]) if isinstance(output, tuple) else h_out

        self._handle = self.module.register_forward_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def diagnostics(self) -> dict:
        return {
            "steered_rows": self._rows,
            "applied_delta_norm_mean": self._delta_sum / self._rows if self._rows else 0.0,
            "applied_delta_norm_max": self._delta_max,
        }


class CachedFLASSteerModel(SteerModel):
    """Inference-only: fit() is unavailable (base SteerModel.fit's "does not
    train" error is exactly right here, not overridden). Persistence is
    `from_pretrained(hf_repo)`, not save()/load() -- the artifact IS the
    published checkpoint, re-fetched deterministically every time."""

    config_cls = CachedFLASConfig

    def __init__(self, config: CachedFLASConfig):
        super().__init__(config)
        self.flow_fn = None
        self.concept_enc = None
        self._revision = None

    @classmethod
    def from_pretrained(cls, hf_repo: str, revision: str | None = None) -> "CachedFLASSteerModel":
        entry = FLAS_CHECKPOINTS.get(hf_repo)
        if entry is None:
            raise FLASCheckpointError(
                f"{hf_repo!r} is not a published FLAS checkpoint; known repos: "
                f"{sorted(FLAS_CHECKPOINTS)}")
        config = cls.config_cls(
            subject=GeneratorRef(entry["model_id"]), layer=entry["layer"], hf_repo=hf_repo,
            model_type=entry["model_type"], hidden_size=entry["hidden_size"],
            num_blocks=entry["num_blocks"], n_steps=entry["n_steps"],
            disable_self_attn=entry["disable_self_attn"],
            disable_cross_attn=entry.get("disable_cross_attn", False),
            disable_mlp=entry.get("disable_mlp", False))
        model = cls(config)
        model._revision = revision
        return model

    def bind(self, model, tokenizer=None):
        super().bind(model, tokenizer)
        self._check_subject()
        return self

    def _check_subject(self):
        from .flow_model import get_text_config
        from ..training_helpers import hidden_size as subject_hidden_size

        model, _ = self._require_bound()
        model_type = get_text_config(model.config).model_type
        if model_type != self.config.model_type:
            raise FLASCheckpointError(
                f"{self.config.hf_repo!r} expects a {self.config.model_type!r} subject "
                f"({self.config.subject.model_id}); bound model is {model_type!r}")
        hd = subject_hidden_size(model, self._layer_module())
        if hd != self.config.hidden_size:
            raise FLASCheckpointError(
                f"{self.config.hf_repo!r} expects hidden_size {self.config.hidden_size}; "
                f"bound model has {hd}")

    def _ensure_loaded(self):
        if self.flow_fn is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from .flow_model import ConceptEncoder, FlowFunction, get_text_config

        model, _ = self._require_bound()
        self._check_subject()
        text_config = get_text_config(model.config)

        fname = f"{self.config.hf_repo.split('/')[-1]}.safetensors"
        path = hf_hub_download(self.config.hf_repo, fname, revision=self._revision)
        state_dict = load_file(path, device="cpu")

        flow_fn = FlowFunction(text_config, num_blocks=self.config.num_blocks, time_conditioned=True,
                               layer_idx=self.config.layer, disable_cross_attn=self.config.disable_cross_attn,
                               disable_self_attn=self.config.disable_self_attn, disable_mlp=self.config.disable_mlp)
        target_dtype = next(iter(state_dict.values())).dtype
        flow_fn.to(target_dtype)
        flow_fn.load_state_dict(state_dict, strict=True)
        flow_fn.to(model.device).eval()
        for p in flow_fn.parameters():
            p.requires_grad = False

        self.flow_fn = flow_fn
        self.concept_enc = ConceptEncoder.from_base_model(model, num_layers=2)

    def _encode_concept(self, concept: Concept):
        import torch

        _, tokenizer = self._require_bound()
        text = concept.description or concept.name
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.concept_max_len)
        device = next(self.concept_enc.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            hidden = self.concept_enc(input_ids, attention_mask)
        flow_dtype = next(self.flow_fn.parameters()).dtype
        return hidden.to(flow_dtype), attention_mask.float()

    def context(self, concept: Concept, T: float = 1.0, n_steps: int | None = None) -> CachedFLASContext:
        self._ensure_loaded()
        concept_hidden, concept_mask = self._encode_concept(concept)
        return CachedFLASContext(self._layer_module(), self.flow_fn, concept_hidden, concept_mask,
                                 T, n_steps if n_steps is not None else self.config.n_steps)
