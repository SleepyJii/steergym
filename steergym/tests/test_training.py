#!/usr/bin/env python3
# Offline tests for the torch-free surface of steergym.axbench.protocol,
# steergym.training_helpers and steergym.reps.models's pure functions -- no
# torch/transformers on PYTHONPATH (this suite is the proof that these are
# importable and exercisable without them: only the fit_* loops themselves
# reach for torch, lazily). Fake tokenizer doubles stand in for HF tokenizers
# (apply_chat_template/decode/__call__) the same way test_helpers.py fakes
# the generator/steer model. Run:
#   PYTHONPATH=<pylibs> python3 steergym/tests/test_training.py
import sys
from dataclasses import dataclass
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from steergym.axbench import AxbenchRowConfig, DiffMeanConfig
from steergym.axbench import protocol
from steergym.concept import Concept
from steergym.generator import GeneratorRef
from steergym.reps.models import (
    LsReFTConfig,
    PreferenceVectorConfig,
    scaled_simpo_terms,
    topk_mean_and_l1,
)
from steergym.training_helpers import linear_decay_lr

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


SUBJECT = GeneratorRef("fake/subject")


# ---------------------------------------------------------------- config defaults (provenance)
dm = DiffMeanConfig(subject=SUBJECT)
check("DiffMeanConfig: n_epochs=1", dm.n_epochs == 1)
check("DiffMeanConfig: batch_size=6", dm.batch_size == 6)
check("DiffMeanConfig: max_length=1024", dm.max_length == 1024)
check("DiffMeanConfig: seed=42", dm.seed == 42)
check("DiffMeanConfig: is an AxbenchRowConfig", isinstance(dm, AxbenchRowConfig))

ls = LsReFTConfig(subject=SUBJECT)
check("LsReFTConfig: lr=0.01", ls.lr == 0.01)
check("LsReFTConfig: weight_decay=0.0", ls.weight_decay == 0.0)
check("LsReFTConfig: n_epochs=3", ls.n_epochs == 3)
check("LsReFTConfig: batch_size=6", ls.batch_size == 6)
check("LsReFTConfig: gradient_accumulation=1", ls.gradient_accumulation == 1)
check("LsReFTConfig: topk=8", ls.topk == 8)
check("LsReFTConfig: coeff_latent_l1_loss=0.005", ls.coeff_latent_l1_loss == 0.005)
check("LsReFTConfig: max_length=1024", ls.max_length == 1024)
check("LsReFTConfig: seed=42", ls.seed == 42)

pv = PreferenceVectorConfig(subject=SUBJECT)
check("PreferenceVectorConfig: lr=0.04", pv.lr == 0.04)
check("PreferenceVectorConfig: n_epochs=18 (canonical)", pv.n_epochs == 18)
check("PreferenceVectorConfig: batch_size=6", pv.batch_size == 6)
check("PreferenceVectorConfig: 10 canonical steering factors",
      tuple(pv.steering_factors) == (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0))
check("PreferenceVectorConfig: preference_pairs=(orig_add, orig_sub)",
      tuple(pv.preference_pairs) == ("orig_add", "orig_sub"))
check("PreferenceVectorConfig: substraction_type=null_it_out", pv.substraction_type == "null_it_out")
check("PreferenceVectorConfig: max_length=512 (the 512-vs-1024 inconsistency)", pv.max_length == 512)
check("PreferenceVectorConfig: seed=42", pv.seed == 42)


# ---------------------------------------------------------------- scaled_simpo_terms
# hand-computed reference: ref_logratios_reverse = ref_lp_l - ref_lp_w = -1.0 - (-2.0) = 1.0
# max(1.0, 1.0) = 1.0; scaled_chosen = (1.0/5) * (-2.0) = -0.4; scaled_rejected = (1.0/4) * (-1.0) = -0.25
# x = -0.4 - (-0.25) = -0.15; loss = -log(sigmoid(-0.15)) = log(1 + exp(0.15))
import math
x = -0.15
expected = -x + math.log1p(math.exp(x))
got = scaled_simpo_terms(ref_lp_w=-2.0, ref_lp_l=-1.0, pol_lp_w=-2.0, pol_lp_l=-1.0,
                         len_w=5.0, len_l=4.0)
check("scaled_simpo_terms: matches hand computation", abs(got - expected) < 1e-12)

# ref_logratios_reverse < 1 -> clamped to 1 (max(x,1))
got2 = scaled_simpo_terms(ref_lp_w=-1.0, ref_lp_l=-1.5, pol_lp_w=-3.0, pol_lp_l=-2.0,
                          len_w=3.0, len_l=2.0)
# ref_logratios_reverse = -1.5-(-1.0) = -0.5 -> max(-0.5,1)=1
scaled_chosen = (1.0 / 3.0) * -3.0
scaled_rejected = (1.0 / 2.0) * -2.0
xx = scaled_chosen - scaled_rejected
expected2 = -xx + math.log1p(math.exp(xx)) if xx < 0 else math.log1p(math.exp(-xx))
check("scaled_simpo_terms: clamps ref_logratios_reverse to 1.0 when below it",
      abs(got2 - expected2) < 1e-12)

# symmetry sanity: identical winning/losing policy+ref logps with equal lengths -> loss = log(2)
# (scaled_chosen == scaled_rejected -> x=0 -> -logsigmoid(0) = log(2))
got3 = scaled_simpo_terms(ref_lp_w=-1.0, ref_lp_l=-1.0, pol_lp_w=-1.0, pol_lp_l=-1.0,
                          len_w=4.0, len_l=4.0)
check("scaled_simpo_terms: equal chosen/rejected scaled logps -> log(2)",
      abs(got3 - math.log(2.0)) < 1e-9)

check("scaled_simpo_terms: always non-negative (it's a -logsigmoid)",
      all(scaled_simpo_terms(a, b, c, d, 3.0, 5.0) >= 0
          for a, b, c, d in [(-1, -2, -1, -2), (-5, -1, -0.5, -3), (-2, -2, -2, -2)]))


# ---------------------------------------------------------------- topk_mean_and_l1
mag, l1 = topk_mean_and_l1([1.0, 5.0, 2.0, 8.0, 0.5], k=2)
check("topk_mean_and_l1: top-2 of [1,5,2,8,0.5] are {8,5} -> mean 6.5", mag == 6.5)
check("topk_mean_and_l1: non-top-2 mean = (1+2+0+0.5)/5",
      abs(l1 - (1.0 + 0.0 + 2.0 + 0.0 + 0.5) / 5) < 1e-12)

mag_all, l1_all = topk_mean_and_l1([3.0, 4.0], k=8)  # k clamped to n=2
check("topk_mean_and_l1: k clamped to n (min(k,n)) -> mean of all values", mag_all == 3.5)
check("topk_mean_and_l1: k clamped to n -> every position is top-k -> l1 mean = 0", l1_all == 0.0)

mag_empty, l1_empty = topk_mean_and_l1([], k=8)
check("topk_mean_and_l1: empty positions -> (0.0, 0.0)", (mag_empty, l1_empty) == (0.0, 0.0))

mag_tie, l1_tie = topk_mean_and_l1([2.0, 2.0, 2.0], k=1)
check("topk_mean_and_l1: ties -> top-1 mean is still one of the tied values", mag_tie == 2.0)
check("topk_mean_and_l1: ties -> l1 mean over the other two", abs(l1_tie - 4.0 / 3) < 1e-12)


# ---------------------------------------------------------------- linear_decay_lr
check("linear_decay_lr: step 0 -> full base_lr", linear_decay_lr(0.01, 0, 100) == 0.01)
check("linear_decay_lr: last step -> base_lr/total_steps",
      abs(linear_decay_lr(0.01, 99, 100) - 0.01 * (100 - 99) / 100) < 1e-15)
check("linear_decay_lr: step == total_steps -> 0.0", linear_decay_lr(0.01, 100, 100) == 0.0)
check("linear_decay_lr: never negative past total_steps",
      linear_decay_lr(0.01, 500, 100) == 0.0)
check("linear_decay_lr: total_steps<=0 -> base_lr (degenerate guard)",
      linear_decay_lr(0.01, 0, 0) == 0.01)


# ---------------------------------------------------------------- fake tokenizer doubles
class FakeTokenizer:
    """Simulates apply_chat_template/decode/__call__ well enough to exercise
    binarize_text without transformers installed. Template shape:
    [BOS, PREFIX, <content-token>, SUFFIX0, SUFFIX1] for a bare user turn;
    add_generation_prompt appends a GEN token; an assistant turn appends the
    assistant content token before the suffix."""

    BOS = 999

    def __init__(self, batch_encoding=False):
        self._batch_encoding = batch_encoding  # simulate arena 5.x BatchEncoding return

    def _content_token(self, content):
        return {"1": 1, "2": 2}.get(content, hash(content) % 90 + 10)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [self.BOS, 100]
        for m in messages:
            ids.append(self._content_token(m["content"]))
            if m["role"] == "assistant":
                ids.append(777)  # assistant-turn marker
        if add_generation_prompt:
            ids.append(200)
        ids.extend([300, 400])  # trailing turn-closing suffix (suffix_length=2)
        if not tokenize:
            return "<BOS>" + "|".join(str(i) for i in ids[1:])
        if self._batch_encoding:
            class _BE(dict):
                pass
            return _BE(input_ids=[ids])
        return ids

    def decode(self, ids):
        return "|".join(str(i) for i in ids)

    def __call__(self, text, max_length=None, truncation=None, return_tensors=None):
        toks = [ord(c) for c in text[:max_length]] if max_length else [ord(c) for c in text]
        return {"input_ids": toks}

    bos_token = "<BOS>"


for batch_encoding in (False, True):
    tok = FakeTokenizer(batch_encoding=batch_encoding)
    label = "BatchEncoding" if batch_encoding else "list[int]"

    ids = protocol.template_ids(tok, [{"role": "user", "content": "1"}])
    check(f"template_ids ({label}): flat list of ints", ids == [999, 100, 1, 300, 400])

    prefix_length, suffix_length = protocol.prefix_suffix_lengths(tok)
    check(f"prefix_suffix_lengths ({label}): prefix_length=2 (divergence at content token)",
          prefix_length == 2)
    check(f"prefix_suffix_lengths ({label}): suffix_length=2 (common trailing [300,400])",
          suffix_length == 2)


    latent_df = pl.DataFrame({
        "concept_id": [7],
        "prompt": ["hello"],
        "response": ["world"],
    })
    texts = protocol.latent_texts(latent_df, tok, suffix_length)
    # messages=[user hello, assistant world] -> ids = [BOS,100, content('hello'), content('world'),
    # 777, 300, 400]; [1:-2] strips BOS and the trailing suffix -> [100, content('hello'),
    # content('world'), 777]
    h = tok._content_token("hello")
    w = tok._content_token("world")
    check(f"latent_texts ({label}): BOS + trailing suffix stripped before decode",
          texts == [f"100|{h}|{w}|777"])

    prompt = protocol.chat_prompt_text(tok, "hello")
    check(f"chat_prompt_text ({label}): BOS stripped, add_generation_prompt applied",
          prompt == f"100|{h}|200|300|400")

    dtext = protocol.binarize_text(tok, "hello", "world", suffix_length)
    # add_generation_prompt=True (unlike latent_texts) -> a trailing GEN(200) token before
    # the stripped suffix
    check(f"binarize_text ({label}): [1:-suffix_length] of the [user,assistant] template, "
          "add_generation_prompt=True",
          dtext == f"100|{h}|{w}|777|200")


# ---------------------------------------------------------------- positive_and_negative_rows
# canonical samples schema: the native axbench columns ride in `meta`
samples = pl.DataFrame({
    "concept_id": [5, 5, -1, -1, -1, -1],
    "prompt": ["a", "b", "c", "d", "e", "f"],
    "response": ["A", "B", "C", "D", "E", "F"],
}).with_columns(pl.struct(
    pl.Series("output_concept", ["cats", "cats", "empty", "empty", "empty", "empty"]),
    pl.Series("category", ["positive", "positive", "negative", "negative", "negative", "negative"]),
    pl.Series("concept_genre", ["news", "news", "news", "news", "sports", "sports"]),
).alias("meta"))
cats = Concept(id=5, name="cats", description="")
positive, negative = protocol.positive_and_negative_rows(samples, cats)
check("positive_and_negative_rows: 2 positives for the matching concept", positive.height == 2)
check("positive_and_negative_rows: negatives genre-matched to the positives' own genre (news)",
      negative.height == 2 and set(negative.get_column("prompt").to_list()) == {"c", "d"})

missing = Concept(id=999, name="nope", description="")
pos_missing, neg_missing = protocol.positive_and_negative_rows(samples, missing)
check("positive_and_negative_rows: unknown concept -> 0 positives, 0 negatives (no genre to match)",
      pos_missing.height == 0 and neg_missing.height == 0)


# ---- hidden_size (duck-typed model/layer doubles; no torch) ----
from steergym.training_helpers import hidden_size


class _Param:
    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)


class _Layer:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


class _CfgModel:
    class config:
        hidden_size = 2304


class _BareModel:
    pass


check("hidden_size: config.hidden_size wins",
      hidden_size(_CfgModel(), _Layer([])) == 2304)
check("hidden_size: no config -> trailing dim of first matrix param (1D params skipped)",
      hidden_size(_BareModel(), _Layer([_Param((640,)), _Param((640, 512))])) == 512)
try:
    hidden_size(_BareModel(), _Layer([_Param((640,))]))
    check("hidden_size: no config, no matrix params -> ValueError", False)
except ValueError:
    check("hidden_size: no config, no matrix params -> ValueError", True)


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
