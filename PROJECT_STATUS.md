# VLM Magic/Mentalism Research Project Status

## Overall Status

**READY FOR FIRST BASELINE**

- Overall: `PASS`
- First real baseline ready: `YES`
- Reason: 1 approved gold example(s), real mp4(s) present, CUDA available, and Qwen cache/load evidence found.
- Generated: 2026-09-03T03:58:31+00:00

## Pipeline

- `PASS` **Repository architecture** (evidence 3) — Package layout, docs, and configs present.
- `PASS` **Environment setup** (evidence 3) — Torch with CUDA available.
- `PASS` **Reproducibility / configuration** (evidence 3) — ExperimentConfig / initialize_experiment available.
- `PASS` **Dataset schema** (evidence 3) — Schema load ok (4 records).
- `PARTIAL` **Dataset validation** (evidence 3) — Toy manifest validates structurally; media checks report missing files (findings=5).
- `PASS` **Video preprocessing** (evidence 4) — Frame select/shuffle smoke ok; 12 real mp4(s) on disk.
- `PASS` **VLM model loading** (evidence 4) — Stub and real Qwen load succeeded.
- `PARTIAL` **VLM video inference** (evidence 3) — Weights/CUDA/mp4 present; full video→answer path not executed in this audit.
- `PARTIAL` **Zero-shot baseline** (evidence 3) — Stub baseline ok and hardware/cache look ready — confirm a real Qwen run separately.
- `PARTIAL` **Baseline evaluation** (evidence 2) — Evaluation helpers exist; no real baseline metrics produced by this audit.
- `PARTIAL` **Failure analysis** (evidence 2) — Analysis module present; not exercised on a real baseline run.
- `PASS` **Preference schema** (evidence 2) — Preference schema/module importable; no human preference labels collected.
- `PARTIAL` **Preference annotation workflow** (evidence 2) — Annotation workflow code exists; requires human judgments later.
- `PARTIAL` **Preference validation** (evidence 2) — Preference validation entry exists; not run on real annotations.
- `PARTIAL` **Bradley-Terry reward model** (evidence 2) — BT reward-model code present; real training not audited here.
- `BLOCKED` **DPO** (evidence 2) — DPO stack code exists; real Qwen DPO needs CUDA + weights + preferences.
- `PASS` **Reward interface** (evidence 3) — hidden_state_exact_match good/bad smoke passed.
- `PARTIAL` **Temporal shuffle** (evidence 3) — Temporal shuffle applied in index smoke; real-video diagnostic not run.
- `PARTIAL` **Temporal / causal reward** (evidence 2) — temporal_iou / causal annotation path exists; not run on real causal labels.
- `PARTIAL` **Common experiment runner** (evidence 2) — Common runner module present; not re-dispatched in this audit beyond stub baseline.
- `BLOCKED` **GRPO** (evidence 2) — GRPO code exists; real VLM GRPO blocked without CUDA/weights/rewards data.
- `PARTIAL` **Comparative evaluation** (evidence 2) — Comparison module present; synthetic fixtures only unless real runs exist.
- `PARTIAL` **Reward-hacking analysis** (evidence 2) — Reward-hacking diagnostics code present; needs before/after artifacts.
- `PASS` **Research reporting** (evidence 3) — ReportConfig + build_experiment_report smoke on experiment_report_toy.

## What Works

- Repository architecture: Package layout, docs, and configs present.
- Environment setup: Torch with CUDA available.
- Reproducibility / configuration: ExperimentConfig / initialize_experiment available.
- Dataset schema: Schema load ok (4 records).
- Video preprocessing: Frame select/shuffle smoke ok; 12 real mp4(s) on disk.
- VLM model loading: Stub and real Qwen load succeeded.
- Preference schema: Preference schema/module importable; no human preference labels collected.
- Reward interface: hidden_state_exact_match good/bad smoke passed.
- Research reporting: ReportConfig + build_experiment_report smoke on experiment_report_toy.

## What Does Not Work / Blocked

- [BLOCKED] DPO: DPO stack code exists; real Qwen DPO needs CUDA + weights + preferences.
- [BLOCKED] GRPO: GRPO code exists; real VLM GRPO blocked without CUDA/weights/rewards data.

## Human Input Required

### I need to provide this now

- (none)

### I need to provide this later

1. **What:** Human preference judgments (A/B explanations)
   - Where: data/annotations/ or preference JSONL
   - Format: PreferencePair schema (docs/PREFERENCE_SCHEMA.md)
   - After: `magic-vlm-validate-preferences --input <prefs.jsonl>`
2. **What:** Causal / temporal span annotations for advanced rewards
   - Where: Manifest causal/temporal fields
   - Format: TemporalSpan + CausalAnnotation on ExampleRecord
   - After: `magic-vlm-compare-objective --manifest ... --predictions ...`
3. **What:** S6 is approved gold. Leave S7 PENDING. Source 4 more hidden-state clip(s) for a 5-clip pilot. Do not gold-label Wikimedia clips.
   - Where: reports/hidden_state_candidates/index.html and HUMAN_INPUT_REQUIRED.md
   - Format: Human decision on pending proposals only; no unverified ground_truth
   - After: `magic-vlm-baseline --config configs/baseline_qwen25vl_3b.yaml --run-id baseline-real-v1 --load-frames`

### Optional

1. **What:** Hugging Face token if gated assets are used
   - Where: Environment variable HF_TOKEN / huggingface-cli login
   - Format: Access token string
   - After: `huggingface-cli whoami`

## Environment Requirements

- Python 3.11.9; torch 2.13.0+cu130 cuda=True
- Real mp4 count: 12
- Qwen HF cache present: True
- GPU available (nvidia-smi/torch): True

## Runtime checks

- GPU_AVAILABLE: `True`
- CUDA_AVAILABLE: `True`
- QWEN_WEIGHTS_AVAILABLE: `True`
- REAL_VLM_LOAD: `True`
- REAL_VIDEO_INFERENCE: `True`
- FIRST_BASELINE_READY: `True`
- readiness_status: `READY_FOR_REAL_BASELINE`

## Hidden-state dataset

- hidden_state_candidates: `7`
- approved_gold_examples: `1`
- pending_review: `1`
- rejected: `10`
- clips_needed: `4`

### WIKIMEDIA CONTROLS

- candidate_count: `5`
- eligible_count: `0`
- pending_human_review: `0`
- rejected_count: `5`

### MAC KING CANDIDATES

- candidate_count: `7`
- eligible_count: `1`
- pending_human_review: `1`
- rejected_count: `5`

### HIDDEN-STATE GOLD

- eligible_count: `1`
- pending_human_review: `1`
- clips_needed_for_pilot: `4`

## First Research Experiment Readiness

`YES` — 1 approved gold example(s), real mp4(s) present, CUDA available, and Qwen cache/load evidence found.

## Next Actions

1. magic-vlm-baseline --config configs/baseline_qwen25vl_3b.yaml --run-id baseline-real-v1 --load-frames
