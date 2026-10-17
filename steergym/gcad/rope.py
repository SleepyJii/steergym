"""Generic (architecture-agnostic) RoPE application + GQA key/value repeat.

Implemented locally instead of upstream's own
`from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb,
repeat_kv`: both functions are copy-identical across every HF decoder that
uses rotate-half RoPE + grouped-query attention (Llama/Qwen2/Mistral/
Gemma2/...), so importing one architecture's module path would tie this
port to that path existing, for no behavioral gain. Verified bit-exact
against the real `transformers` package's own copies in
tests/test_gcad.py's parity leg. Torch imported lazily (module-level stays
stdlib-only -- `import steergym` must work in a bare venv)."""


def rotate_half(x):
    import torch

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """q, k: [..., H, T, D] (head dim at `unsqueeze_dim`); cos, sin:
    [..., T, D] -- HF's own default calling convention (unsqueeze_dim=1
    broadcasts cos/sin across the head axis)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep: int):
    """[B, num_kv_heads, T, D] -> [B, num_kv_heads * n_rep, T, D]."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
