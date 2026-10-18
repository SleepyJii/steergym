# Papers

Source papers for the methods and benchmarks implemented in steergym. Each
subpackage README records the upstream repo/commit and all adaptation deltas.

## AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders
(Wu 2025) — [arXiv:2501.17148](https://arxiv.org/abs/2501.17148)
- `LsReFTModel` (ReFT-r1): rank-1 rows, TopKReLU training, addition steering
- `DiffMeanModel`: difference-of-means rows (closed form)
- `AxbenchPromptSteerModel`: the paper's prompt-steering arm
- `Concept500Benchmark`: concepts, AlpacaEval steering prompts, factor × max-act dosing, latent max-act pass
- `AxbenchConceptEval` / `AxbenchInstructEval` / `AxbenchFluencyEval`: the leaderboard's gpt-4o-mini 0-2 judges, verbatim templates; strict-harmonic-mean aggregate via `lm_judge_rating`
- released dictionaries `pyvene/gemma-{reft-r1,diffmean}-2b-it-res` (verification anchors)

## Improved Representation Steering for Language Models (RePS)
(Wu 2025) — [arXiv:2505.20809](https://arxiv.org/abs/2505.20809)
- `PreferenceVectorModel`: rank-1 rows trained with scaled-SimPO over orig_add / orig_sub preference pairs (null-it-out subtraction arm)

## ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment
(ICLR 2026) — [github.com/ZhaoHongjue/odesteer](https://github.com/ZhaoHongjue/odesteer)
- `ODESteerModel` / `StepODESteerModel`: kernel-logistic density-ratio velocity field over count-sketch features, fixed-step integration
- bundled baselines, re-implemented per the odesteer source: `CAAModel` (Rimsky et al., ACL 2024), `RepEModel` (Zou et al. 2023), `ITIProbeModel`, `LinActModel`, `MiMiCModel`

## Activation Transport (AcT)
(Rodriguez et al.) — [arXiv:2410.23054](https://arxiv.org/abs/2410.23054), code: apple/ml-act
- `MeanAcTModel`, `LinearAcTModel`: sequential multi-module transport maps (mean shift / 1-D sort-based linear OT with support masking)

## PID Steering
(Nguyen et al.) — [arXiv:2510.04309](https://arxiv.org/abs/2510.04309), code: dungnvnus/pid-steering (a Mean-AcT fork)
- `PIDAcTModel` ("PID-AcT" is this library's alias for the method): per-coordinate PID law (canonical paper form + a `repo_compat` mode reproducing the released code's degenerate law)

## Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control (A-LQR)
— [arXiv:2604.19018](https://arxiv.org/abs/2604.19018)
- `ALQRModel` / `ALQRContext`: per-layer time-varying LQR feedback tracking a contrastive setpoint (rank-1 gain rows, no resident d×d gains)
- `SPIDModel`: the paper's own S-PID controller ablation (code-pinned; the paper ships no closed-form equation)

## Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions (GCAD)
— [arXiv:2605.10664](https://arxiv.org/abs/2605.10664)
- `GCADModel` / `GCADContext`: gated cropped attention-delta, per-token demand-gated steering of attention outputs

## Beyond Steering Vector: Flow-based Activation Steering (FLAS)
(NeurIPS 2026) — [arXiv:2605.05892](https://arxiv.org/abs/2605.05892)
- `CachedFLASSteerModel`: inference-only loading of the authors' published FlowBlock checkpoints; flowtime `T` dosing

## Persona Vectors: Monitoring and Controlling Character Traits in Language Models
(Chen 2025) — [arXiv:2507.21509](https://arxiv.org/abs/2507.21509)
- `PersonaVectorModel`: zero-train trait rows (unit or raw convention)
- `personavec.TraitEval` / `CoherenceEval`: the paper's own 0-100 logprob-aggregated API judges, verbatim prompts
- `PersonaVectorsBenchmark`: the 7 published traits as steering concepts, upstream rollout protocol

## Instruction-Following Evaluation for Large Language Models (IFEval)
(Zhou 2023) — [arXiv:2311.07911](https://arxiv.org/abs/2311.07911)
- `IFEvalEval`: official vendored checkers, strict + loose variants

## Improving Instruction-Following in Language Models through Activation Steering
(Stolfo 2024) — [arXiv:2410.12877](https://arxiv.org/abs/2410.12877)
- `StolfoDiffMeanModel`: last-prompt-token diff-mean instruction vectors (precomputed rows)
- `StolfoPerplexityEval`: their perplexity degeneracy gate
- `StolfoQualityEval`: their two-stage LM quality judge (App. F)
- `IFEvalBenchmark`: their augmented IFEval pool (vendored), one concept per instruction type
