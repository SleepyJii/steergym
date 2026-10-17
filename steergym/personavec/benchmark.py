import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..axbench.benchmark import LATENT_META_KEYS, TRAIN_META_KEYS, canonical_samples
from ..benchmark import LABEL_NEG, LABEL_POS, META_LABEL, BenchmarkConfig, PolarBenchmark, SteerBenchmark
from ..concept import Concept
from ..generator import GenParams
from ..steer import GeneratorRef
from .traits import TRAITS, VENDOR_DIR, build_packs, fold, system_text

FILTERS = ("openai", "none")


@dataclass
class PersonaVectorsConfig(BenchmarkConfig):
    subject: GeneratorRef              # generates extraction data AND is steered
    dump_dir: str | None = None        # gen_traits-style dump: cached source, or the generative stage's cache target
    trait_data_dir: str | None = None  # trait JSONs override; default = the vendored copy
    traits: tuple = TRAITS
    n_extract: int = 20                # extraction questions per trait (x 5 instruction pairs); 20 = ALL, upstream
    n_eval: int = 20                   # held-out eval questions per trait; 20 = ALL, upstream
    n_per_question: int = 10           # sampled responses per (question x instruction) prompt, both arms
    extraction_temperature: float = 1.0  # upstream couples these: T=1.0 for n>1, T=0.0 (greedy) iff n=1 --
                                       # kept explicit here so deviations are recorded config; a faithful
                                       # single-sample run is n=1/T=0.0 (eval_persona.py main())
    seed: int = 42
    gen_tokens: int = 256              # config of record (upstream: up to 1000)
    gen_batch: int = 16
    latent_n: int = 36
    filter: str = "openai"             # effective-pair filter: openai (the paper's own judges
                                       # at their 50/100 thresholds) | none


class PersonaVectorsBenchmark(SteerBenchmark, PolarBenchmark):
    """persona_vectors traits as steering concepts (Chen et al. 2025): one
    concept per trait, Concept.name = the trait word (what the axbench fits
    filter on), Concept.description = the rubric's definition paragraph (what
    the faithful TraitEval judges against).

    Fully generative, like IFEval: the subject answers extraction questions
    under pos/neg-instructed folded prompts and plain, an effective-pair
    filter (config.filter) keeps pairs where the pos arm expresses the trait
    and the neg arm doesn't (both coherent), and those rows feed everything —
    axbench train/dpo/latent formats plus extraction/extraction_data.parquet,
    the input to PersonaVectorModel.fit (the response-avg resid diff moved
    there per the native-training doctrine). Cached mode reads a
    gen_traits-layout dump; dumps from earlier pipelines lack the extraction
    parquet, so extraction_samples() (and fit) need a dump generated here —
    everything else is interchangeable. filter="openai" needs a real
    OPENAI_API_KEY when the generative stage runs."""

    config_cls = PersonaVectorsConfig

    def __init__(self, config: PersonaVectorsConfig):
        super().__init__(config)
        if config.filter not in FILTERS:
            raise ValueError(f"filter {config.filter!r} must be one of {FILTERS}")
        self._generator = None
        self._packs = None

    def bind_generator(self, generator):
        self._generator = generator
        return self

    def _trait_data_dir(self):
        return Path(self.config.trait_data_dir) if self.config.trait_data_dir else VENDOR_DIR

    def _dump(self) -> Path | None:
        return Path(self.config.dump_dir) if self.config.dump_dir else None

    def _dump_populated(self) -> bool:
        dump = self._dump()
        return dump is not None and (dump / "generate" / "train_data.parquet").exists()

    def packs(self) -> list[dict]:
        if self._packs is None:
            if self._dump_populated():
                self._packs = json.loads((self._dump() / "packs.json").read_text())
            else:
                self._packs = build_packs(self.config.traits, self.config.n_extract,
                                          self.config.n_eval, self._trait_data_dir())
        return self._packs

    def concepts(self) -> list[Concept]:
        return [Concept(id=p["concept_id"], name=p["trait"], description=p["description"])
                for p in self.packs()]

    def eval_samples(self) -> pl.DataFrame:
        # one row per held-out question. Upstream's steering eval SAMPLES
        # n_per_question (reference default 10, T=1.0) responses per
        # question -- that repetition is the scoring runscript's knob
        # (repeat rows / rerun arms), not benchmark data
        rows = [{"concept_id": p["concept_id"], "prompt": q, "response": None,
                 "meta": {"trait": p["trait"]}}
                for p in self.packs() for q in p["questions_eval"]]
        return pl.DataFrame(rows)

    def train_samples(self) -> pl.DataFrame:
        self._require_dump("train_samples")
        native = pl.read_parquet(self._dump() / "generate" / "train_data.parquet")
        return canonical_samples(native, TRAIN_META_KEYS)

    def latent_samples(self) -> pl.DataFrame:
        self._require_dump("latent_samples")
        native = pl.read_parquet(self._dump() / "pregen" / "latent_eval_data.parquet")
        return canonical_samples(native, LATENT_META_KEYS)

    def polar_train_samples(self) -> pl.DataFrame:
        self._require_dump("polar_train_samples")
        native = pl.read_parquet(self._dump() / "generate" / "dpo_train_data.parquet")
        label = (pl.when(pl.col("category") == "positive").then(pl.lit(LABEL_POS))
                 .otherwise(pl.lit(LABEL_NEG)))
        return native.select(
            "concept_id",
            pl.col("input").alias("prompt"),
            pl.col("winning_output").alias("winning"),
            pl.col("losing_output").alias("losing"),
            pl.struct(label.alias(META_LABEL), *[pl.col(k) for k in TRAIN_META_KEYS]).alias("meta"),
        )

    def extraction_samples(self) -> pl.DataFrame:
        # the pos/neg/plain extraction rows with per-row filter scores --
        # PersonaVectorModel.fit consumes the effective pos/neg pairs; absent
        # from dumps written by earlier pipelines (this adapter's superset artifact)
        self._require_dump("extraction_samples")
        path = self._dump() / "extraction" / "extraction_data.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing: this dump predates the steergym adapter (earlier "
                "pipelines never persisted extraction rows) -- regenerate the dump here to fit "
                "PersonaVectorModel natively, or load its rows via config.dump_dir")
        return pl.read_parquet(path)

    def evals(self) -> list:
        # the paper's own judges: 0-100 logprob-aggregated gpt-4.1-mini
        # trait/coherence scoring, verbatim prompts (real API spend)
        from .evals import CoherenceEval, TraitEval
        return [TraitEval(), CoherenceEval()]

    # ---------------------------------------------------------------- generative stage
    def _require_dump(self, method):
        if self._dump_populated():
            return
        dump = self._dump()
        if dump is None:
            raise RuntimeError(
                f"PersonaVectorsBenchmark.{method} needs a generated dump and no dump_dir "
                "is set -- point dump_dir at a gen_traits-style dump, or set dump_dir AND "
                "bind_generator(GeneratorLM) to generate one there")
        if self._generator is None:
            raise RuntimeError(
                f"PersonaVectorsBenchmark.{method}: dump_dir {str(dump)!r} has no "
                "generate/train_data.parquet and no generator is bound -- "
                "bind_generator(GeneratorLM) to generate the pack there")
        if self._generator.ref.model_id != self.config.subject.model_id:
            raise RuntimeError(
                f"bound generator is {self._generator.ref.model_id!r} but config.subject "
                f"is {self.config.subject.model_id!r} -- extraction responses must come "
                "from the subject")
        self._generate_dump()

    def _filter_evals(self):
        """(trait_eval, coherence_eval, scale, threshold) for config.filter;
        None for filter='none'. openai = the faithful upstream judges at
        their own thresholds."""
        if self.config.filter == "none":
            return None
        from .evals import CoherenceEval, TraitEval
        return TraitEval(), CoherenceEval(), 100.0, 50.0

    def _generate_dump(self):
        # gen_traits.generate, minus PersonaVector weights (now
        # PersonaVectorModel.fit over extraction_samples()) and resp/ CSVs
        # (superseded by extraction/extraction_data.parquet). Same
        # axbench-idiom layout otherwise.
        from transformers import set_seed

        cfg = self.config
        dump = self._dump()
        gen_dir, pregen_dir, ext_dir = dump / "generate", dump / "pregen", dump / "extraction"
        for d in (gen_dir, pregen_dir, ext_dir):
            d.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed)
        gen = GenParams(max_new_tokens=cfg.gen_tokens, temperature=cfg.extraction_temperature,
                        do_sample=cfg.extraction_temperature > 0, batch_size=cfg.gen_batch, seed=None)
        filter_evals = self._filter_evals()
        packs = self.packs()
        concepts = {c.id: c for c in self.concepts()}

        metadata_lines, train_rows, dpo_rows, latent_rows, ext_rows = [], [], [], [], []
        filter_stats = {}
        for p in packs:
            cid, trait = p["concept_id"], p["trait"]
            questions = p["questions_extract"]

            # folded prompts: each question x 5 instruction pairs x
            # n_per_question sampled rollouts -> pos/neg (upstream's
            # Question.get_input(n_per_question) expansion; the effective
            # filter pairs pos/neg by aligned row = (question, instruction,
            # sample), exactly their aligned-CSV mask); one plain prompt per
            # question (the plain arm feeds the contrastive fits, not upstream
            # protocol -- it stays single-rollout)
            pos_texts, neg_texts, pair_index = [], [], []
            for qi, q in enumerate(questions):
                for ii, instr in enumerate(p["instruction"]):
                    for si in range(cfg.n_per_question):
                        pos_texts.append(fold(system_text(trait, instr["pos"], "pos"), q))
                        neg_texts.append(fold(system_text(trait, instr["neg"], "neg"), q))
                        pair_index.append((qi, ii, si))
            pos_resps = self._generator.generate(
                [self._generator.chat_prompt(t) for t in pos_texts], gen)
            neg_resps = self._generator.generate(
                [self._generator.chat_prompt(t) for t in neg_texts], gen)
            plain_resps = self._generator.generate(
                [self._generator.chat_prompt(q) for q in questions], gen)
            pair_questions = [questions[qi] for qi, _, _ in pair_index]
            plain_by_q = dict(zip(questions, plain_resps))

            def scored(prompts, resps):
                if filter_evals is None:
                    return [None] * len(resps), [None] * len(resps)
                trait_eval, coh_eval, _, _ = filter_evals
                frame = pl.DataFrame({"concept_id": [cid] * len(resps),
                                      "prompt": prompts, "response": resps})
                t = next(iter(trait_eval.score(frame, concepts).values()))
                c = next(iter(coh_eval.score(frame, concepts).values()))
                return t, c

            pos_trait, pos_coh = scored(pair_questions, pos_resps)
            neg_trait, neg_coh = scored(pair_questions, neg_resps)
            plain_trait, plain_coh = scored(questions, plain_resps)

            # effective PAIRED mask, generate_vec.py get_persona_effective
            # semantics: pos expresses (>= thr), neg doesn't (< scale - thr),
            # both coherent (>= scale/2); < 3 effective -> fall back to ALL
            if filter_evals is None:
                effective = [True] * len(pair_questions)
            else:
                _, _, scale, thr = filter_evals
                effective = [
                    (pt is not None and nt is not None and pc is not None and nc is not None
                     and pt >= thr and nt < (scale - thr) and pc >= scale / 2 and nc >= scale / 2)
                    for pt, nt, pc, nc in zip(pos_trait, neg_trait, pos_coh, neg_coh)]
            n_effective = sum(effective)
            fallback_used = n_effective < 3
            if fallback_used:
                effective = [True] * len(pair_questions)
            filter_stats[trait] = {
                "n_pairs": len(pair_questions), "n_effective": n_effective,
                "fallback_used": fallback_used,
                "mean_pos_trait": _mean(pos_trait), "mean_neg_trait": _mean(neg_trait),
                "mean_plain_trait": _mean(plain_trait), "mean_pos_coh": _mean(pos_coh),
                "mean_neg_coh": _mean(neg_coh), "mean_plain_coh": _mean(plain_coh),
            }

            for arm, texts, resps, tr, coh in (
                    (LABEL_POS, pos_texts, pos_resps, pos_trait, pos_coh),
                    (LABEL_NEG, neg_texts, neg_resps, neg_trait, neg_coh)):
                for i, (text, resp) in enumerate(zip(texts, resps)):
                    qi, ii, si = pair_index[i]
                    ext_rows.append({
                        "concept_id": cid, "prompt": text, "response": resp,
                        "meta": {META_LABEL: arm, "question": questions[qi],
                                 "instruction_idx": ii, "sample_idx": si,
                                 "trait_score": tr[i],
                                 "coherence": coh[i], "effective": effective[i]}})
            for qi, q in enumerate(questions):
                ext_rows.append({
                    "concept_id": cid, "prompt": q, "response": plain_by_q[q],
                    "meta": {META_LABEL: "plain", "question": q, "instruction_idx": None,
                             "sample_idx": None,
                             "trait_score": plain_trait[qi], "coherence": plain_coh[qi],
                             "effective": None}})

            eff_q = [q for q, e in zip(pair_questions, effective) if e]
            eff_pos = [r for r, e in zip(pos_resps, effective) if e]
            common = {"output_concept": trait, "concept_genre": "text",
                      "dataset_category": "instruction", "concept_id": cid}
            for q, r in zip(eff_q, eff_pos):
                train_rows.append({"input": q, "output": r, "category": "positive", **common})
                dpo_rows.append({"input": q, "winning_output": r,
                                 "losing_output": plain_by_q[q], "category": "positive", **common})
            # global -1 pool: ALL plain AND all neg-instructed responses
            # (input = the plain question in both cases, per the dump spec)
            for q, r in list(zip(questions, plain_resps)) + list(zip(pair_questions, neg_resps)):
                train_rows.append({"input": q, "output": r, "output_concept": "EEEEE",
                                   "concept_genre": "text", "category": "negative",
                                   "dataset_category": "instruction", "concept_id": -1})
            for q, r in zip(eff_q[:cfg.latent_n], eff_pos[:cfg.latent_n]):
                latent_rows.append({"input": q, "output": r, "category": "positive",
                                    "sae_link": trait, "sae_id": 0, **common})
            for q in questions[:cfg.latent_n]:
                latent_rows.append({"input": q, "output": plain_by_q[q], "category": "negative",
                                    "sae_link": trait, "sae_id": 0, **common})
            metadata_lines.append(json.dumps({
                "concept_id": cid, "concept": trait, "ref": f"personavec/{trait}/{cid}",
                "concept_genres_map": {trait: ["text"]}}))

        pl.DataFrame(train_rows).write_parquet(gen_dir / "train_data.parquet")
        pl.DataFrame(dpo_rows).write_parquet(gen_dir / "dpo_train_data.parquet")
        pl.DataFrame(latent_rows).write_parquet(pregen_dir / "latent_eval_data.parquet")
        pl.DataFrame(ext_rows).write_parquet(ext_dir / "extraction_data.parquet")
        (gen_dir / "metadata.jsonl").write_text("\n".join(metadata_lines) + "\n")
        (dump / "packs.json").write_text(json.dumps(packs, indent=2))
        (dump / "filter_stats.json").write_text(json.dumps(filter_stats, indent=2))


def _mean(xs):
    vals = [x for x in xs if x is not None]
    return round(sum(vals) / len(vals), 4) if vals else None
