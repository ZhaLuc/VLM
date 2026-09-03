# Professor Demo Guide — Visual Walkthrough

Use this in a short meeting. Each screen is one idea. Full detail lives in
[`PROJECT_COMPLETE_GUIDE.md`](PROJECT_COMPLETE_GUIDE.md). One-page sheet:
[`PROFESSOR_XU_ONE_PAGE.md`](PROFESSOR_XU_ONE_PAGE.md).

Open diagrams from `docs/assets/` or the homepage [`index.html`](index.html).

---

## Slide/Screen 1 — Project Question

**Show:** [`assets/project_overview.svg`](assets/project_overview.svg)

**Talking point:**

> I'm testing whether post-training can make a VLM better at reasoning about
> hidden states and mechanisms in magic videos — not just captioning what is
> already visible.

---

## Slide/Screen 2 — The Concrete Task

**Show:** [`assets/hidden_state_task.svg`](assets/hidden_state_task.svg)

**Talking point:**

> The model sees a short Mac King clip with an apparent hand transfer and no
> final reveal. It has to say which hand still has the coin.

---

## Slide/Screen 3 — What the Model Did

**Show:**

```text
Question:
Which hand contains the coin after the apparent transfer?

Model:
right

Ground truth:
right
```

**Evidence:** `reports/real_zero_shot_baseline/` · run `baseline-real-v1` ·
model `Qwen/Qwen2.5-VL-3B-Instruct` · device `cuda:0`.

**Talking point:**

> This is a real Qwen2.5-VL-3B inference on a real MP4, not a mock. The pipeline
> from decode → frames → generate → exact-match eval worked end-to-end.

---

## Slide/Screen 4 — Why This Was Not Enough

**Show:**

```text
n = 1
accuracy = 1.0
```

**Talking point:**

> One correct example is not evidence of general hidden-state reasoning,
> temporal reasoning, causal reasoning, or that post-training would help.

---

## Slide/Screen 5 — Intended Training Pipeline

**Show:** [`assets/rlhf_pipeline.svg`](assets/rlhf_pipeline.svg) and
[`assets/completed_vs_planned.svg`](assets/completed_vs_planned.svg)

```text
VLM
 ↓
candidate answers
 ↓
human preferences / objective reward
 ↓
DPO / GRPO
 ↓
independent evaluation
```

**Talking point:**

> The next stage would have tested whether preference- or reward-based
> post-training actually changes behavior on held-out clips.

---

## Slide/Screen 6 — Why GRPO

**Show:** [`assets/grpo_pipeline.svg`](assets/grpo_pipeline.svg)

**Talking point:**

> For a rule-based hidden-state reward, GRPO samples several answers for one
> prompt and updates from relative advantages inside that group — no separate
> critic like PPO.

---

## Slide/Screen 7 — Evaluation Design

**Show:** [`assets/dataset_and_evaluation.svg`](assets/dataset_and_evaluation.svg)
and [`assets/temporal_shuffle.svg`](assets/temporal_shuffle.svg)

```text
seen trick
unseen trick
unseen performer
unseen camera
unseen wording
ordered vs shuffled
```

**Talking point:**

> Evaluation is designed to catch memorization and shortcuts — including
> transparent-cup leakage we already rejected from gold.

---

## Slide/Screen 8 — Final Status

**Show:** [`assets/professor_talking_points.svg`](assets/professor_talking_points.svg)

```text
REAL VLM PIPELINE: COMPLETE

POST-TRAINING STUDY: NOT COMPLETED

STATUS: PAUSED
```

**Talking point:**

> I stopped here because of time and dataset constraints. The core zero-shot
> prototype is real and reproducible; the learning experiment remains future work.

---

## Optional backup screens

| If asked… | Open |
|-----------|------|
| What is a VLM? | `assets/vlm_concept.svg` |
| Why DPO? | `assets/dpo_pipeline.svg` |
| Why not PPO first? | `assets/ppo_pipeline.svg` |
| Reward hacking? | `assets/reward_hacking.svg` |
| Project health | `../PROJECT_STATUS.md` |
| Candidate review | `../reports/hidden_state_candidates/index.html` |
