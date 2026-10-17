import math


def linear_decay_lr(base_lr: float, step: int, total_steps: int) -> float:
    """transformers get_scheduler('linear', num_warmup_steps=0) formula."""
    if total_steps <= 0:
        return base_lr
    return base_lr * max(0.0, (total_steps - step) / total_steps)


def hidden_size(model, layer_module) -> int:
    """Residual width of the subject model, for the rank-1 nn.Linear(hd, 1)
    the ReFT-family fits build. HF configs carry it; test doubles may not, so
    fall back to the trailing dimension of the layer module's first matrix
    parameter."""
    size = getattr(getattr(model, "config", None), "hidden_size", None)
    if size is not None:
        return int(size)
    matrices = [p for p in layer_module.parameters() if p.dim() >= 2]
    if not matrices:
        raise ValueError("cannot infer the subject's hidden size: no config.hidden_size "
                         "and the layer module has no matrix parameters")
    return int(matrices[0].shape[-1])


# ---------------------------------------------------------------- seeded, padded batching
def pad_stack(rows, pad_id, dtype, torch):
    max_len = max(len(r) for r in rows)
    out = torch.full((len(rows), max_len), pad_id, dtype=dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r if torch.is_tensor(r) else torch.tensor(r, dtype=dtype)
    return out


class _SeededLMBatches:
    """DataLoader-equivalent over pre-tokenized rows: one torch.Generator,
    seeded once, reshuffled (advancing, not resetting) each time this object
    is iterated -- matching axbench's Model.make_dataloader (a fresh
    generator per concept, persistent across that concept's epochs)."""

    def __init__(self, tokenizer, rows, prefix_length, batch_size, seed):
        self._tokenizer = tokenizer
        self._rows = rows
        self._prefix_length = prefix_length
        self._batch_size = batch_size
        import torch
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return math.ceil(len(self._rows) / self._batch_size)

    def __iter__(self):
        import torch
        n = len(self._rows)
        order = torch.randperm(n, generator=self._generator).tolist() if n else []
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        for i in range(0, n, self._batch_size):
            idx = order[i:i + self._batch_size]
            batch_rows = [self._rows[j] for j in idx]
            input_ids = pad_stack([r["input_ids"] for r in batch_rows], pad_id, torch.long, torch)
            lengths = torch.tensor([len(r["input_ids"]) for r in batch_rows], dtype=torch.long)
            T = input_ids.shape[1]
            attention_mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)).long()
            labels = torch.full_like(input_ids, -100)
            for b, r in enumerate(batch_rows):
                seq_len = len(r["input_ids"])
                pl_ = r["prompt_length"]
                labels[b, pl_:seq_len] = input_ids[b, pl_:seq_len]
            pos = torch.arange(T).unsqueeze(0)
            interv_mask = (pos >= self._prefix_length) & (pos < lengths.unsqueeze(1))
            yield {
                "input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "interv_mask": interv_mask,
            }


def lm_train_batches(tokenizer, rows, prefix_length, max_length, batch_size, seed):
    """Seeded-shuffle DataLoader-equivalent over pre-tokenized causal-LM rows
    (each a dict with 'input_ids' [1D LongTensor, already truncated to
    max_length by the caller] and 'prompt_length' [int]). Right-pads with
    the tokenizer's pad token; builds labels (-100 outside [prompt_length,
    seq_len)) and an intervention-position mask ([prefix_length, seq_len)).
    """
    del max_length  # truncation happens at tokenization time, before rows is built
    return _SeededLMBatches(tokenizer, rows, prefix_length, batch_size, seed)


def tokenize_row(tokenizer, prompt: str, output: str, max_length: int, torch):
    prompt_ids = tokenizer(prompt, max_length=max_length, truncation=True)["input_ids"]
    full_ids = tokenizer(prompt + output, max_length=max_length, truncation=True)["input_ids"]
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "prompt_length": len(prompt_ids),
    }
