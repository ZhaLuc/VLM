# Professor Xu - One Page

**Status:** PAUSED - zero-shot prototype complete; post-training not completed.

### Research question

Can preference- and/or reward-based post-training improve an open VLM’s ability to
infer **hidden states** and explain mechanisms in magic/mentalism videos, while
generalizing beyond a small training set rather than exploiting shortcuts?

### Backbone

Gated spine: validated JSONL examples -> frame sampling -> VLM generate -> exact-match eval -> archived artifacts. Training (DPO/GRPO) was meant to plug into that spine after more gold + preferences.

### What I built

A reproducible research codebase: dataset schema + validation + leakage checks,
video frame sampling, Qwen2.5-VL loading/inference, zero-shot baseline runner,
project-health audits, and scaffolds for preferences / Bradley-Terry RM / DPO /
GRPO / temporal shuffle / reward-hacking analysis
(`src/magic_vlm/`, `configs/`, `docs/`).

### What worked

Real CUDA stack (RTX 3060, PyTorch `2.13.0+cu130`), real
`Qwen/Qwen2.5-VL-3B-Instruct` load, real Mac King `Movie6.MP4` preprocessing,
one human-approved gold example (`mac_king_s006`), formal zero-shot run
`baseline-real-v1` / `REAL_ZERO_SHOT_BASELINE`.

### Main result

Question: *Which hand contains the coin after the apparent transfer?*  
Model: **right** · Ground truth: **right** · Correct · **n = 1**, accuracy **1.0**.  
Evidence: `reports/real_zero_shot_baseline/`.

### Why it stopped

Time + dataset scale: only **1/5** approved gold clips; no preference labels;
therefore no honest DPO/GRPO experiment. Not a software failure.

### What I would do next

1. Approve ~4 more hidden-state clips (leave S7 pending until reviewed).  
2. Collect preference pairs for explanations.  
3. Run DPO (explanations) and GRPO (objective hidden-state reward), then
   independent held-out + temporal-shuffle eval.

### One GRPO detail

Same prompt → multiple sampled answers → rewards → **group-normalized**
advantages (relative to that group’s mean/std), then policy update - no separate
critic (DeepSeekMath).

### One DPO detail

Preference pairs (chosen/rejected) update the policy **directly** without a
separate reward model or PPO loop (Rafailov et al.).

### One evaluation concern

**Reward hacking / shortcuts:** proxy reward can rise while true reasoning does
not; need held-out tricks, leakage controls, and independent metrics.

### One question for Professor Xu

Given n=1 zero-shot success, would you prioritize expanding the gold set first,
or collecting a small preference set for a DPO smoke before a larger GRPO study?
