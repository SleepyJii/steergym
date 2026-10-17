"""Batched layer-activation extraction: the machinery shared by every
activation-space fit (DiffMean's pooled token means, the odesteer vector
pack's last-token reads) — right-padded forwards over pre-encoded ids with a
capture hook on one layer module."""


def batched_layer_activations(model, layer_module, encoded, batch_size, pad_id):
    """Yield (h [B, T, d] on CPU, lengths [B]) per batch_size chunk of
    `encoded` (lists of token ids), in order."""
    import torch

    from .training_helpers import pad_stack

    with torch.no_grad():
        for i in range(0, len(encoded), batch_size):
            batch_ids = [torch.tensor(e, dtype=torch.long) for e in encoded[i:i + batch_size]]
            if not batch_ids:
                continue
            input_ids = pad_stack(batch_ids, pad_id, torch.long, torch)
            lengths = torch.tensor([len(x) for x in batch_ids], dtype=torch.long)
            T = input_ids.shape[1]
            attention_mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)).long()

            captured = {}

            def hook(mod, args, output, captured=captured):
                captured["h"] = output[0] if isinstance(output, tuple) else output

            handle = layer_module.register_forward_hook(hook)
            model(input_ids=input_ids.to(model.device),
                  attention_mask=attention_mask.to(model.device))
            handle.remove()
            yield captured["h"].cpu(), lengths


def last_token_activations(generator, texts, config, log=lambda msg: None):
    """[N, d] float32 CPU: the layer-config.layer activation at each text's
    final token. Order-preserving."""
    import torch

    tokenizer = generator.tokenizer
    tokenizer.padding_side = "right"
    layer_module = generator.model.model.layers[config.layer]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    encoded = [tokenizer(t, max_length=config.max_length, truncation=True)["input_ids"]
               for t in texts]
    out = [h[torch.arange(len(lengths)), lengths - 1]
           for h, lengths in batched_layer_activations(
               generator.model, layer_module, encoded, config.batch_size, pad_id)]
    return torch.cat(out, dim=0).float()
