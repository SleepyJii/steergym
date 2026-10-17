#!/usr/bin/env python3
# Offline tests for GeneratorLM.generate kwargs plumbing + load() pad-token
# fallback. NEEDS torch (CPU is enough). Run:
#   PYTHONPATH=<pylibs>:<torchlibs> python3 steergym/tests/test_generator.py
# transformers is replaced in sys.modules for the load() checks (saved and
# restored, so an arena env's real module is put back untouched).
import sys
import types as pytypes
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from steergym import GenParams, GeneratorLM
from steergym.steer import GeneratorRef

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


# ---------------------------------------------------------------- fakes
class FakeTokenizer:
    """Prompt 'p<i>' -> ids [10i, 10i+1, 10i+2]; batch_decode renders the
    actual token values it is handed, so the decode checks pin the
    prompt-length slice and cross-batch response ordering, not just counts."""

    def __init__(self, pad_token=None, eos_token="</s>"):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.padding_side = "right"  # generate() must flip this to left

    def __call__(self, chunk, return_tensors="pt", padding=True, truncation=True):
        ids = torch.tensor([[10 * int(p[1:]) + k for k in range(3)] for p in chunk])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def batch_decode(self, out, skip_special_tokens=True):
        return ["dec:" + ",".join(str(int(x)) for x in row) for row in out]


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.generate_kwargs = []  # one dict per model.generate call

    def generate(self, **kwargs):
        self.generate_kwargs.append(kwargs)
        ids = kwargs["input_ids"]
        return torch.cat([ids, ids[:, :2] + 1000], dim=1)


def run(gen_params, n_prompts=3):
    model, tok = FakeModel(), FakeTokenizer(pad_token="<pad>")
    glm = GeneratorLM(GeneratorRef(model_id="fake"), model, tok)
    responses = glm.generate([f"p{i}" for i in range(n_prompts)], gen_params)
    return model, tok, responses


# ---------------------------------------------------------------- kwargs pass-through
model, tok, responses = run(GenParams(batch_size=2))
check("unset knobs: model.generate receives exactly the pre-existing kwargs, values included",
      all(set(kw) == {"input_ids", "attention_mask", "max_new_tokens",
                      "do_sample", "temperature"}
          and kw["max_new_tokens"] == 128 and kw["do_sample"] is True
          and kw["temperature"] == 1.0 for kw in model.generate_kwargs))
check("batching unchanged (3 prompts @ batch 2 -> 2 calls)",
      len(model.generate_kwargs) == 2
      and [kw["input_ids"].shape[0] for kw in model.generate_kwargs] == [2, 1])
check("decode slices the prompt off and keeps cross-batch response order",
      responses == [f"dec:{1000 + 10 * i},{1001 + 10 * i}" for i in range(3)])
check("padding_side forced left", tok.padding_side == "left")

model, _, _ = run(GenParams(batch_size=8, top_p=0.9, repetition_penalty=1.1,
                            no_repeat_ngram_size=2))
kw = model.generate_kwargs[0]
check("all three knobs set: passed with their values",
      {"top_p", "repetition_penalty", "no_repeat_ngram_size"} <= set(kw)
      and kw["top_p"] == 0.9 and kw["repetition_penalty"] == 1.1
      and kw["no_repeat_ngram_size"] == 2)

model, _, _ = run(GenParams(batch_size=8, repetition_penalty=1.1))
check("partial set: only the set knob is passed",
      "repetition_penalty" in model.generate_kwargs[0]
      and "top_p" not in model.generate_kwargs[0]
      and "no_repeat_ngram_size" not in model.generate_kwargs[0])

# falsy-but-set values are real replay inputs (odesteer's own defaults are
# top_p=1.0 / repetition_penalty=1.0 / no_repeat_ngram_size=0, and explicit-0
# differs from unset once the model ships its own generation_config.json) —
# the filter must be `is not None`, never truthiness
model, _, _ = run(GenParams(batch_size=8, top_p=1.0, repetition_penalty=1.0,
                            no_repeat_ngram_size=0))
kw = model.generate_kwargs[0]
check("falsy-but-set knobs (0 / 1.0) still pass through explicitly",
      kw.get("top_p") == 1.0 and kw.get("repetition_penalty") == 1.0
      and kw.get("no_repeat_ngram_size") == 0)

check("GenParams knob defaults are all None (bit-unchanged by default)",
      GenParams().top_p is None and GenParams().repetition_penalty is None
      and GenParams().no_repeat_ngram_size is None)


# ---------------------------------------------------------------- load() pad fallback
class FakeConfig:
    def __init__(self):
        self.eos_token_id = 7
        self.pad_token_id = None


class FakeHFModel:
    def __init__(self):
        self.config = FakeConfig()

    def eval(self):
        return self


def stub_transformers(pad_token, eos_token="</s>"):
    hf_model = FakeHFModel()
    hf_tok = FakeTokenizer(pad_token=pad_token, eos_token=eos_token)
    sys.modules["transformers"] = pytypes.SimpleNamespace(
        AutoModelForCausalLM=pytypes.SimpleNamespace(
            from_pretrained=lambda *a, **k: hf_model),
        AutoTokenizer=pytypes.SimpleNamespace(
            from_pretrained=lambda *a, **k: hf_tok),
    )
    return hf_model, hf_tok


_real_transformers = sys.modules.get("transformers")
try:
    hf_model, hf_tok = stub_transformers(pad_token=None)
    glm = GeneratorLM.load(GeneratorRef(model_id="fake"), dtype="float32")
    check("padless tokenizer: pad falls back to eos (load-bearing half) + "
          "config parity-of-record line",
          glm.tokenizer.pad_token == "</s>"
          and glm.model.config.pad_token_id == 7)

    hf_model, hf_tok = stub_transformers(pad_token="<pad>")
    hf_model.config.pad_token_id = 3
    glm = GeneratorLM.load(GeneratorRef(model_id="fake"), dtype="float32")
    check("tokenizer with a pad token: load leaves pad + config untouched",
          glm.tokenizer.pad_token == "<pad>" and glm.model.config.pad_token_id == 3)

    stub_transformers(pad_token=None, eos_token=None)
    try:
        GeneratorLM.load(GeneratorRef(model_id="fake"), dtype="float32")
        check("tokenizer with neither pad nor eos: load raises", False)
    except ValueError:
        check("tokenizer with neither pad nor eos: load raises", True)
finally:
    if _real_transformers is not None:
        sys.modules["transformers"] = _real_transformers
    else:
        sys.modules.pop("transformers", None)


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
