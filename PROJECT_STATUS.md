# VLM Magic/Mentalism Research Project Status

## Overall Status

**NOT READY FOR RESEARCH RUN**

- Overall: `BLOCKED`
- First real baseline ready: `PARTIALLY`
- Reason: Real video present, but first baseline still blocked by: CUDA GPU (torch.cuda), local Qwen2.5-VL weights / HF cache
- Generated: 2026-09-03T01:58:05+00:00

## Pipeline

- `PASS` **Repository architecture** (evidence 3) — Package layout, docs, and configs present.
- `BLOCKED` **Environment setup** (evidence 3) — Torch is CPU-only; real Qwen2.5-VL research runs need CUDA.
- `PASS` **Reproducibility / configuration** (evidence 3) — ExperimentConfig / initialize_experiment available.
- `PASS` **Dataset schema** (evidence 3) — Schema load ok (4 records).
- `PARTIAL` **Dataset validation** (evidence 3) — Toy manifest validates structurally; media checks report missing files (findings=5).
- `PASS` **Video preprocessing** (evidence 4) — Frame select/shuffle smoke ok; 5 real mp4(s) on disk.
- `BLOCKED` **VLM model loading** (evidence 3) — Stub load works; real Qwen load with allow_download=False failed (cache_present=False).
- `BLOCKED` **VLM video inference** (evidence 2) — Real video VLM inference blocked (need real mp4 + CUDA + Qwen weights).
- `BLOCKED` **Zero-shot baseline** (evidence 3) — Stub baseline smoke passed; real hidden-state baseline blocked (no CUDA, no Qwen cache).
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
- Reproducibility / configuration: ExperimentConfig / initialize_experiment available.
- Dataset schema: Schema load ok (4 records).
- Video preprocessing: Frame select/shuffle smoke ok; 5 real mp4(s) on disk.
- Preference schema: Preference schema/module importable; no human preference labels collected.
- Reward interface: hidden_state_exact_match good/bad smoke passed.
- Research reporting: ReportConfig + build_experiment_report smoke on experiment_report_toy.

## What Does Not Work / Blocked

- [BLOCKED] Environment setup: Torch is CPU-only; real Qwen2.5-VL research runs need CUDA.
- [BLOCKED] VLM model loading: Stub load works; real Qwen load with allow_download=False failed (cache_present=False).
- [BLOCKED] VLM video inference: Real video VLM inference blocked (need real mp4 + CUDA + Qwen weights).
- [BLOCKED] Zero-shot baseline: Stub baseline smoke passed; real hidden-state baseline blocked (no CUDA, no Qwen cache).
- [BLOCKED] DPO: DPO stack code exists; real Qwen DPO needs CUDA + weights + preferences.
- [BLOCKED] GRPO: GRPO code exists; real VLM GRPO blocked without CUDA/weights/rewards data.

## Human Input Required

### I need to provide this now

1. **What:** Source 5 hidden-state clips that pass docs/HIDDEN_STATE_ELIGIBILITY.md (opaque concealment). Do not gold-label the Wikimedia transparent-cup pilots.
   - Where: data/videos/ plus provenance; see docs/HIDDEN_STATE_VIDEO_SOURCING_GUIDE.md
   - Format: mp4 + provenance JSON + pending annotations (not gold until approved)
   - After: `Open reports/hidden_state_candidates/index.html and HUMAN_INPUT_REQUIRED.md`
2. **What:** Obtain a CUDA GPU environment and CUDA-enabled PyTorch
   - Where: Training/inference host (not this CPU-only audit host)
   - Format: NVIDIA GPU + matching CUDA torch wheel
   - After: `python -c "import torch; print(torch.cuda.is_available())"`
3. **What:** Place local Qwen2.5-VL weights (3B or 7B Instruct)
   - Where: HF hub cache or a local directory referenced by model_id
   - Format: Transformers checkpoint directory for Qwen2.5-VL
   - After: `magic-vlm-baseline --config configs/baseline_qwen25vl_3b.yaml --load-frames`

### I need to provide this later

1. **What:** Human preference judgments (A/B explanations)
   - Where: data/annotations/ or preference JSONL
   - Format: PreferencePair schema (docs/PREFERENCE_SCHEMA.md)
   - After: `magic-vlm-validate-preferences --input <prefs.jsonl>`
2. **What:** Causal / temporal span annotations for advanced rewards
   - Where: Manifest causal/temporal fields
   - Format: TemporalSpan + CausalAnnotation on ExampleRecord
   - After: `magic-vlm-compare-objective --manifest ... --predictions ...`

### Optional

1. **What:** Hugging Face token if gated assets are used
   - Where: Environment variable HF_TOKEN / huggingface-cli login
   - Format: Access token string
   - After: `huggingface-cli whoami`

## Environment Requirements

- Python 3.11.9; torch 2.13.0+cpu cuda=False
- Real mp4 count: 5
- Qwen HF cache present: False

## First Research Experiment Readiness

`PARTIALLY` — Real video present, but first baseline still blocked by: CUDA GPU (torch.cuda), local Qwen2.5-VL weights / HF cache

## Next Actions

1. GPU host with CUDA-enabled PyTorch for practical Qwen2.5-VL runs
2. Download Qwen2.5-VL locally or point model_id at a local directory
3. Obtain footage that passes docs/HIDDEN_STATE_ELIGIBILITY.md; Wikimedia transparent-cup pilots are controls, not hidden-state gold
4. Re-run: python scripts/project_health.py  after videos + CUDA + Qwen weights
