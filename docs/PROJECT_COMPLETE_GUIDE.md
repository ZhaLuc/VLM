# Complete Project Guide

**Visual Reasoning and Explanation from Magic and Mentalism Demonstrations**

**Status:** PAUSED - zero-shot prototype complete; post-training study not completed.

This guide is the authoritative educational explanation of the repository. It is written for a technically capable student who knows basic machine learning but may not have worked with VLMs, RLHF, reward models, PPO, DPO, or GRPO.

**Honesty rule:** This is a **paused research prototype**, not a completed post-training study. One real zero-shot example succeeded (`n = 1`). That does **not** prove strong reasoning, causal understanding, generalization, or that post-training would help.

| Start here | Path |
|------------|------|
| Visual homepage | [`index.html`](index.html) |
| Professor 2-minute walkthrough | [`PROFESSOR_DEMO_GUIDE.md`](PROFESSOR_DEMO_GUIDE.md) |
| One-page meeting sheet | [`PROFESSOR_XU_ONE_PAGE.md`](PROFESSOR_XU_ONE_PAGE.md) |
| Live project health | [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md) |
| Formal S6 baseline evidence | [`../reports/real_zero_shot_baseline/`](../reports/real_zero_shot_baseline/) |

Diagrams live in [`assets/`](assets/).

---

## 1. Project in One Minute

Imagine a short magic video where a performer appears to toss a coin from one hand to the other, but the coin may still be hidden in the first hand. A **vision-language model (VLM)** watches sampled frames from that video, reads a question in English, and produces a short text answer. Humans (or automatic rules) can then score whether the answer is good. Those scores can later be used to **post-train** the model so future answers improve. Finally, an **independent evaluation** checks whether improvement is real or just shortcut learning.

Conceptual flow:

```text
magic video
   +
question
   ↓
VLM
   ↓
answer
   ↓
feedback / reward
   ↓
post-training
   ↓
better model
   ↓
independent evaluation
```

**Where this repository actually stopped:** the first real VLM **zero-shot baseline** on one human-approved hidden-state clip (Mac King S6). Preference collection, reward-model training, DPO, and GRPO on the real VLM were **not** completed.

---

## 1b. The Backbone: How This System Is Wired End-to-End

If the "one minute" story is the *idea*, this section is the *machine*. Everything in the repo hangs off one spine:

```text
manifest JSONL (examples)
        |
        v
schema + validation + leakage checks
        |
        v
video path + content hash
        |
        v
frame sampling (and optional temporal shuffle)
        |
        v
prompt template + frames + question
        |
        v
model loader (stub or Qwen2.5-VL)
        |
        v
raw text generation
        |
        v
parse / canonicalize answer
        |
        v
exact-match evaluation (+ optional rewards)
        |
        v
run artifacts under runs/  (+ mirrored reports/)
```

That spine is what "worked" on S6. Training methods (DPO/GRPO) are **side branches** that were scaffolded to plug into the same data/eval spine later - they were never the completed experiment.

### Why the project is shaped this way

The scientific fear is not "can a VLM generate text about a video?" (that is easy to demo). The fear is:

1. **Leakage:** the answer is already visible (transparent cups, final reveal).
2. **Shortcut learning:** the model memorizes a performer, camera, or wording.
3. **Proxy gaming:** a training reward goes up while true hidden-state skill does not.
4. **Overclaiming:** n=1 or toy smokes get narrated as "reasoning improved."

So the backbone is deliberately **gates first, training second**:

| Layer | Job | Main code |
|-------|-----|-----------|
| Schema | Force every example to carry identity + provenance + split | `src/magic_vlm/schemas.py`, `dataset.py` |
| Validation | Catch missing fields, bad media, split leakage | `validate.py`, `scripts/validate_dataset.py` |
| Eligibility / human review | Refuse to gold-label leaky clips | `hidden_state_eligibility.py`, inventory + review JSONL |
| Video | Deterministic frame indices from real MP4s | `video.py`, `scripts/sample_frames.py` |
| Experiment config | Freeze seeds, model id, generation, run ids | `experiment.py`, `configs/*.yaml` |
| Models | Load stub or real HF VLM without baking training in | `models.py` |
| Inference | Produce **raw** model text, keep it inspectable | `inference.py` |
| Evaluation | Exact-match vs ground truth; keep parse failures visible | `evaluation.py`, `baseline.py` |
| Rewards (planned training) | Score answers for GRPO / diagnostics | `rewards.py` |
| Preferences (planned training) | Store A/B human judgments for DPO / RM | `preferences.py`, `annotation.py` |
| DPO / GRPO / RM | Training loops (toy/fixture-proven; not real post-training) | `dpo.py`, `grpo.py`, `reward_model.py` |
| Health / reporting | Tell you what is real vs scaffolded | `project_health.py`, `reporting.py` |

### What happens on the one real success (S6)

This is the concrete walk through the backbone for the only approved gold example:

1. **Gold gate.** Human APPROVE recorded for Mac King S6. Only then is a scoreable row written to `data/examples/hidden_state_pilot.jsonl` with:
   - `question`: Which hand contains the coin after the apparent transfer?
   - `ground_truth`: `right`
   - `video.path`: `data/videos/Movie6.MP4`
   - `split`: `held_out`
   - provenance pointing at Cui et al. 2011 / PMC3202226
2. **Config.** `configs/baseline_qwen25vl_3b.yaml` selects that manifest, the Qwen2.5-VL-3B Instruct checkpoint, greedy decoding (`temperature: 0.0`), and frame sampling (`max_frames: 8`, uniform).
3. **Preprocess.** The runner opens the MP4, samples 8 ordered frame indices (formal run used indices like `0, 31, 61, ..., 215`), and feeds **project-sampled frames** into the model - not a mystery closed-source API.
4. **Generate.** Untouched base weights (`untouched_base_3b`) on `cuda:0` (RTX 3060, Torch `2.13.0+cu130`) emit raw text `right`.
5. **Evaluate.** Exact-match: parsed answer `right` == ground truth `right` -> correct. Metrics: `n_examples=1`, `overall_accuracy=1.0`.
6. **Freeze evidence.** Formal artifacts live under `runs/baseline-real-v1/` (local/gitignored) and a committed mirror at `reports/real_zero_shot_baseline/` labeled `REAL_ZERO_SHOT_BASELINE`, kept distinct from the earlier smoke folder.

Command shape:

```bash
python scripts/run_baseline.py --config configs/baseline_qwen25vl_3b.yaml --run-id baseline-real-v1 --load-frames
```

### What "post-training" would attach to this backbone

Nothing mystical - the same prompt/example objects would be reused:

```text
                    +--> human prefs --> DPO  (explanation quality)
baseline examples --+
                    +--> objective reward --> GRPO  (e.g. exact-match hidden state)
                              |
                              v
                     same held_out / leakage / temporal-shuffle eval
```

That is why so much code exists that was never "finished research": the spine had to be real before training claims could be honest. Pausing after the first real zero-shot run is therefore a **data/time** stop, not a missing folder hierarchy.

### Mental model of the directories

Think of three nested folders:

1. **Research question** (hidden-state + explanation under post-training, with generalization).
2. **Engineering spine** (data -> frames -> VLM -> eval -> artifacts) - **this is done for n=1**.
3. **Learning methods** (SFT/DPO/PPO/GRPO/RM) - **implemented as modules, not completed as experiments**.

If you only remember one sentence: *the backbone is a gated multimodal evaluation pipeline; learning algorithms were optional adapters that never got real preference/reward data.*

Diagrams: [`assets/project_overview.svg`](assets/project_overview.svg), [`assets/completed_vs_planned.svg`](assets/completed_vs_planned.svg).

---

## 2. Professor Xu's Research Question

> Can preference- and/or reward-based post-training improve an open VLM's ability to infer hidden states and explain mechanisms in magic/mentalism demonstrations, while generalizing beyond the small training set rather than merely exploiting shortcuts?

This is a **scientific** question, not merely an application demo, because it asks:

1. Whether post-training **changes** multimodal behavior on a hard visual-inference task.
2. Whether gains **transfer** to unseen tricks / performers / cameras / wordings.
3. Whether measured gains reflect **reasoning** rather than reward hacking or leakage.

The planned contribution was a controlled benchmark + post-training comparison (especially DPO for explanations and GRPO for objective hidden-state reward), plus temporal-shuffle and reward-hacking diagnostics - **not** a finished claim that any method already improved reasoning.

---

## 3. Why Magic and Mentalism?

Magic and mentalism demonstrations are useful because they force:

| Skill | Why magic stresses it |
|-------|------------------------|
| Visual reasoning | Props, hands, gaze, and misdirection must be interpreted together |
| Hidden state | The true object location can be occluded after an apparent transfer |
| Temporal reasoning | Order of events matters; the last frame alone may mislead |
| Causal reasoning | Apparent cause (the toss) may not match the true mechanism |
| Explanation | “What happened?” is richer than a one-word label |
| Misdirection | Attention is deliberately steered away from the method |
| Inference from demonstrations | The model must infer what is not explicitly shown |

**This project is NOT:**

- robot manipulation or physically performing tricks
- robotics control
- a magic video game
- a generic chatbot without a grounded visual task

It is a **VLM evaluation and (planned) post-training** study on short demonstration clips.

Source materials used in the repo include Cui et al. (2011) Mac King supplementary videos ([PMC3202226](https://pmc.ncbi.nlm.nih.gov/articles/PMC3202226/)) and Wikimedia / PeerJ cups-and-balls pilot footage retained as **controls**, not gold hidden-state items. See `docs/HIDDEN_STATE_VIDEO_SOURCING_GUIDE.md` and `docs/WIKIMEDIA_CLIP_PREPARATION.md`.

---

## 4. What Is a VLM?

### Building blocks

- **LLM (large language model):** a neural network trained to predict the next text token given previous tokens.
- **Vision model:** a network that turns images/frames into numerical features.
- **Multimodal model:** combines more than one modality (here: vision + language).
- **VLM (vision-language model):** a multimodal model that takes images/video frames **and** text, and typically generates text.

Conceptual flow:

```text
VIDEO / IMAGE
     +
TEXT
     ↓
visual information
+
language information
     ↓
multimodal model
     ↓
text response
```

### Tokens, logits, probabilities, generation (intro)

1. Text is broken into **tokens** (subword pieces).
2. The model produces **logits** (raw scores) for every token in the vocabulary for the *next* position.
3. Logits are converted to **probabilities** (often via softmax).
4. **Generation** repeatedly picks the next token (greedily or by sampling) until a stop condition.

This repository does **not** invent undocumented internal layers of Qwen2.5-VL. Operationally we load `Qwen/Qwen2.5-VL-3B-Instruct` via the project model loader (`src/magic_vlm/models.py`) and run inference (`src/magic_vlm/inference.py`). Model card: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

Diagram: [`assets/vlm_concept.svg`](assets/vlm_concept.svg).

---

# How Does the AI Actually Learn?

## 5. Parameters

A neural network’s **parameters** (weights) are the numbers that define its behavior. Changing parameters changes what answers are likely. Learning usually means updating parameters so better answers become more probable.

## 6. Tokens

A response is a **sequence of tokens**, not a single atomic “thought.” Even a one-word answer like `right` is produced as one or more tokens under the tokenizer. Training methods often increase or decrease the probability of entire answer sequences.

## 7. Logits and probabilities

At each step the model scores every vocabulary token. Higher logit → higher probability after normalization. Greedy decoding (used in the formal S6 baseline: `temperature: 0.0`, `do_sample: false`) always picks the highest-probability next token.

## 8. Loss

**Loss** answers: “How wrong was the model?” Common supervised loss increases when the model assigned low probability to the desired next tokens. Preference/RL methods use different objectives, but the idea remains: a scalar that guides updates.

## 9. Gradient descent

```text
prediction
   ↓
loss
   ↓
gradient
   ↓
parameter update
   ↓
slightly different model
```

Gradients tell each parameter how to move to reduce loss (or improve the RL/preference objective). Many small updates accumulate into noticeable behavior change.

## 10. Pretraining

Before this project, the VLM was already pretrained (and instruction-tuned by its creators) on large multimodal corpora. We start from that **base** checkpoint (`untouched_base_3b` in the baseline report). We did **not** pretrain from scratch.

## 11. SFT (supervised fine-tuning)

```text
video + question + desired answer
              ↓
            model
              ↓
       increase probability
       of desired answer
```

SFT teaches imitation of gold demonstrations. The project considered SFT as an **optional baseline**, but the main planned learning experiments were preference/reward methods (DPO / GRPO). No SFT experiment on the real VLM was completed as the primary study.

## 12. Post-training

**Post-training** here means further adaptation **after** the public pretrained/instruction-tuned checkpoint - using preferences, rewards, or demonstrations - to specialize behavior on the research task. It differs from pretraining in scale, objective, and data: small curated task data and explicit preference/reward signals rather than next-token prediction on a web-scale corpus.

---

# The Actual Research Task

## 13. Hidden-State Reasoning (S6)

Concrete example used in this repository:

```text
coin visible in right hand
          ↓
apparent transfer
          ↓
coin remains hidden
          ↓
video ends
          ↓
"Which hand contains the coin?"
          ↓
"right"
```

Why temporal information matters: if you only classify the **final** frame, gaze and hand pose can suggest the left hand (misdirection). The correct answer depends on earlier events (fake toss retaining the coin in the right hand), documented in Cui et al. 2011 and confirmed by human review for S6.

Diagram: [`assets/hidden_state_task.svg`](assets/hidden_state_task.svg).  
Gold row: `data/examples/hidden_state_pilot.jsonl`.

## 14. Why the First Wikimedia Clips Were Rejected

Early Wikimedia / PeerJ cups-and-balls clips were inspected and labeled **not suitable** for hidden-state gold because of issues such as:

- transparent cups (state still visible)
- late reveal that leaks the answer
- visible final state that turns the task into recognition rather than inference

They were **retained as controls / pipeline smoke**, not discarded from the world - and the team did **not** lower the benchmark standard just to inflate the gold count. That is scientific rigor: a harder empty set is better than a polluted “success.”

See `docs/HIDDEN_STATE_ELIGIBILITY.md`, `data/examples/wikimedia_pilot_review.md`, and `reports/hidden_state_candidates/`.

## 15. Mac King Video Set

Supplementary clips from Cui et al. 2011 (*Front. Hum. Neurosci.*), local files `Movie1.MP4`-`Movie7.MP4`, inventoried in `data/examples/mac_king_review.jsonl` and `reports/mac_king_clip_review/`.

| ID | Paper condition (approx.) | Role in this project |
|----|---------------------------|----------------------|
| S1 | Magic trick (with reveal) | Visible-reveal **control**; revealed counterpart of S6 |
| S2 | Real toss (with reveal) | Real-toss reveal **control**; counterpart of S7 |
| S3 | (related fake-toss / study condition in inventory) | Control / not gold unless separately approved |
| S4 | Reveal-present condition in inventory | Visible-reveal **control** |
| S5 | No-coin fake toss | Ill-posed for “which hand has the coin” - control only |
| **S6** | Magic trick **without** reveal | **First approved hidden-state gold** (`ground_truth: right`) |
| S7 | Real toss **without** reveal | Hidden-state **candidate**; remains **PENDING** |

**Reveal vs no-reveal:** clips that show the empty/full palm at the end leak the answer. No-reveal clips stop before that flourish, so the model must infer the hidden location.

**Why S6 was approved:** human researcher APPROVE after video + paper consistency (fake toss retains coin in right; evaluation clip omits S1 reveal). Decision: `data/examples/human_review_decisions.json`.

**Why S7 stayed pending:** candidate with remaining uncertainty (e.g., whether the cupped left hand already leaks the coin). Not gold until human APPROVE.

---

# Dataset

## 16. Dataset Structure

Each example record (see `docs/DATASET_SCHEMA.md`, `src/magic_vlm/schemas.py`) typically includes:

| Field | Meaning |
|-------|---------|
| `clip_id` / `example_id` | Stable identifiers |
| `trick_id` | Trick family (grouping for splits) |
| `performer_id` | Who performs |
| `camera_id` | Camera / recording setup |
| `question` | Natural-language query |
| `ground_truth` | Target answer for exact-match eval |
| `temporal` / `causal` | Optional span annotations (mostly unused in the real S6 row) |
| `split` | e.g. `held_out` |
| `provenance` | Source URL, license notes, creation metadata |

Metadata matters because **without** trick/performer/camera tags you cannot detect leakage or claim generalization.

## 17. Train/Test Leakage

```text
random split
```

vs

```text
grouped split by trick / performer / setup
```

A random split can put nearly identical clips from the same trick into both train and test. The model may look like it “generalizes” while memorizing a performer, prop layout, or camera angle. Grouped splits and held-out tricks are how this project planned to stress-test that failure mode (`src/magic_vlm/validate.py`, dataset docs).

Diagram: [`assets/dataset_and_evaluation.svg`](assets/dataset_and_evaluation.svg).

---

# Human Feedback and Reward Learning

## 18. Human Preferences

```text
Question:
Which hand contains the coin?

Answer A:
"The coin remains in the right hand."

Answer B:
"The coin transferred to the left hand."

Human:
A is better.
```

**Pairwise preference learning** trains from comparisons (A ≻ B), not only absolute scores. Schema: `docs/PREFERENCE_SCHEMA.md`. **No real human preference dataset was collected** before pause.

## 19. Reward Models

```text
video + question + response
                ↓
          reward model
                ↓
             score
```

A scalar reward is useful because RL algorithms need a number. It is only a **proxy** for true quality and can be wrong (verbosity, confidence, keyword stuffing). Project module: `src/magic_vlm/reward_model.py`, docs `docs/REWARD_MODEL.md`. Real BT training on human prefs: **not completed**.

## 20. Bradley-Terry

Intuition: if humans preferred A over B, the reward model should learn `score(A) > score(B)`.

$$
P(A \text{ beats } B) = \sigma(r_A - r_B)
$$

**Tiny numeric example:** if \(r_A = 1.0\) and \(r_B = 0.0\), then \(r_A - r_B = 1\), \(\sigma(1) \approx 0.73\). The model is trained so preferred answers get higher \(r\).

---

# Reinforcement Learning

## 21. What Is Reinforcement Learning?

```text
agent
 ↓
action
 ↓
environment
 ↓
reward
 ↓
learning
```

In this project (language, not robotics):

```text
agent/policy = VLM

action = generated token sequence / answer

reward = correctness or preference-based quality signal
```

The “environment” is the prompt (video frames + question) plus a scorer. There is no robot arm.

## 22. RLHF

```text
base model
   ↓
candidate answers
   ↓
human preferences
   ↓
reward model
   ↓
RL optimization
   ↓
updated model
```

Classic pipeline popularized by InstructGPT / RLHF: preferences → reward model → RL (often PPO) with KL regularization against a reference policy. See Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT). Diagram: [`assets/rlhf_pipeline.svg`](assets/rlhf_pipeline.svg).

## 23. PPO

**Proximal Policy Optimization** updates the policy using reward (or advantage estimates) while clipping updates so the new policy does not move too far too fast.

Concepts:

- **Policy:** the VLM that generates answers
- **Reward:** scalar quality
- **Advantage:** how much better an action was than expected
- **Value / critic:** a model estimating expected return (extra memory)
- **KL regularization:** stay close to a reference model
- **Policy update:** gradient step on the clipped PPO objective

**Why not the starting point here:** PPO typically needs policy + reference + reward (+ critic). On a 3B VLM with limited GPU memory (RTX 3060 12GB), that stack is heavy. The project deferred PPO and prioritized DPO / GRPO. Diagram: [`assets/ppo_pipeline.svg`](assets/ppo_pipeline.svg). Hugging Face [TRL](https://huggingface.co/docs/trl) documents PPO trainers; this repo did not complete real PPO training.

---

# GRPO

## 24. GRPO

**Group Relative Policy Optimization** (DeepSeekMath) samples multiple answers for **one** prompt, scores them, and uses **relative** advantages within that group - without a separate critic.

```text
             one prompt
                 |
        +--------+--------+
        |        |        |
      answer   answer   answer ...
        |        |        |
      reward   reward   reward
        \        |       /
         \       |      /
          group comparison
                 ↓
          relative advantage
                 ↓
             model update
```

**Numeric example (from the research plan):**

Rewards: `0.9, 0.7, 0.2, 0.1`  
Mean: `0.475`  
Population std ≈ `0.3345`  
Advantages ≈ `+1.27, +0.67, -0.82, -1.12`

Relative advantages up-weight better-than-group answers and down-weight worse ones. Attractive here because a **rule-based** hidden-state reward (`hidden_state_exact_match` in `src/magic_vlm/rewards.py`) can score candidates without a learned RM.

Cite: DeepSeekMath / GRPO ([DeepSeekMath paper](https://arxiv.org/abs/2402.03300)). Repo docs: `docs/GRPO.md`, code `src/magic_vlm/grpo.py`, script `scripts/train_grpo.py`. **Real VLM GRPO training: not completed.**

Diagram: [`assets/grpo_pipeline.svg`](assets/grpo_pipeline.svg).

---

# DPO

## 25. DPO

```text
video + question
      +
chosen answer
      +
rejected answer
          ↓
        DPO
          ↓
     updated model
```

**Direct Preference Optimization** (Rafailov et al.) fits preferences **directly** in the policy, without a separate reward model and without a conventional PPO loop. It fitted the **explanation** task: humans can mark which explanation is better.

Cite: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290). Repo: `docs/DPO.md`, `src/magic_vlm/dpo.py`, `scripts/train_dpo.py`. **Real DPO training: not completed** (no real preference dataset).

Diagram: [`assets/dpo_pipeline.svg`](assets/dpo_pipeline.svg).

---

# Compare Methods

## 26. SFT vs DPO vs PPO vs GRPO

| Method | Main signal | Reward model | Online sampling | Critic | Intended project role |
| ------ | ----------- | ------------ | --------------- | ------ | --------------------- |
| SFT | Gold demonstrations | No | No | No | Optional baseline |
| DPO | Preference pairs | No | No | No | Explanation task |
| PPO | Reward | Usually | Yes | Yes | Deferred |
| GRPO | Objective/preference reward | Optional | Yes | No separate critic | Hidden-state/temporal task |

**Plain English:**

- **SFT:** copy good answers.
- **DPO:** learn from “this answer beats that one.”
- **PPO:** sample, score with a reward, update carefully; needs more moving parts.
- **GRPO:** sample a group, compare within the group, update; good fit for exact-match hidden-state rewards.

---

# Temporal and Causal Reasoning

## 27. Temporal Reasoning

```text
ordered:
1 → 2 → 3 → 4

shuffled:
3 → 1 → 4 → 2
```

Same frames; different order. If accuracy collapses under shuffle, the model was using order (or at least was sensitive to it). If accuracy is unchanged, it may be ignoring temporal structure or using non-temporal shortcuts.

Implementation: `src/magic_vlm/temporal.py`, `scripts/run_temporal_shuffle.py`, `docs/TEMPORAL_SHUFFLE.md`. Real post-training temporal study: **not completed** (index/preprocess smoke exists).

Diagram: [`assets/temporal_shuffle.svg`](assets/temporal_shuffle.svg).

## 28. Causal Reasoning

```text
correlation
vs
temporal dependence
vs
causation
```

Magic makes causal annotation hard: the **apparent** cause (the toss) is designed to disagree with the **true** mechanism (retention / palm). The repo scaffolds causal/temporal reward fields (`docs/TEMPORAL_CAUSAL_REWARD.md`) but did not complete a causal post-training experiment.

---

# Reward Hacking

## 29. Reward Hacking

```text
reward:
0.40 → 0.90

true task accuracy:
50% → 51%
```

> The model learned how to satisfy the evaluator better than it learned the underlying task.

Failure modes discussed in the plan / `docs/REWARD_HACKING.md`:

- verbosity bias
- confidence bias
- keyword stuffing
- answer-distribution shortcuts
- parser exploitation
- visual leakage (transparent cups, reveals)

Independent evaluation is essential: held-out tricks, controls, and metrics that are not identical to the training reward.

Diagram: [`assets/reward_hacking.svg`](assets/reward_hacking.svg).

---

# What Was Actually Built

## 30. Repository Walkthrough

```text
repository
├── configs/          # YAML for baseline, DPO, GRPO, rewards, reports
├── data/
│   ├── examples/     # manifests, reviews, gold pilot JSONL
│   ├── annotations/  # preference annotation landing zone
│   ├── provenance/   # source provenance JSON
│   └── videos/       # local MP4s (gitignored; not committed)
├── docs/             # architecture + this teaching archive
│   ├── assets/       # SVG diagrams
│   ├── index.html    # visual homepage
│   └── ...
├── reports/          # committed audit + baseline evidence mirrors
├── runs/             # local experiment outputs (gitignored)
├── scripts/          # CLI wrappers
├── src/magic_vlm/    # Python package
├── tests/            # automated tests
├── PROJECT_STATUS.md # regenerated health banner
├── README.md         # entry point
└── magic-vlm-research-plan-v2.md
```

**Major package modules (exist):** `dataset`, `validate`, `video`, `models`, `inference`, `evaluation`, `baseline`, `preferences`, `reward_model`, `rewards`, `dpo`, `grpo`, `temporal`, `reward_hacking`, `comparison`, `reporting`, `project_health`, etc. under `src/magic_vlm/`.

Do not assume every script has been run on real large-scale data; many training paths are **scaffolded** and tested on toys/fixtures only.

## 31. Actual System Pipeline

**Completed path:**

```text
Data (JSONL manifests)
 ↓
Validation / leakage checks
 ↓
Video preprocessing (frame sample / optional shuffle)
 ↓
VLM inference (Qwen2.5-VL-3B)
 ↓
Evaluation (exact match)
 ↓
Experiment artifacts (runs/ + reports/)
 ↓
Reporting / project health
```

**Unfinished training branch:**

```text
Preferences → Reward Model → DPO
                            \
                             → evaluation

Objective Reward → GRPO
                    \
                     → temporal experiments
```

Diagram: [`assets/completed_vs_planned.svg`](assets/completed_vs_planned.svg).

---

# What Actually Worked

## 32. Verified Milestones

| Component | Status | Evidence |
|-----------|--------|----------|
| Repository infrastructure | PASS | this repository |
| Tests | PASS | `pytest -q` (see latest run in FINAL STATUS after regen) |
| Video decode | PASS | real MP4s under `data/videos/` (local) |
| Frame sampling | PASS | baseline preprocessing on Movie6 |
| CUDA | PASS | RTX 3060; `torch 2.13.0+cu130`; `docs/CUDA_ENVIRONMENT.md` |
| Qwen2.5-VL-3B load | PASS | HF cache; `src/magic_vlm/models.py` |
| Real video inference | PASS | S6 / Movie6 |
| Human-approved gold example | PASS | `data/examples/hidden_state_pilot.jsonl` |
| Zero-shot baseline | PASS | `reports/real_zero_shot_baseline/` (`n=1`) |
| 5-clip benchmark | INCOMPLETE | only 1 approved |
| Human preferences | NOT COMPLETED | no real dataset |
| DPO | NOT COMPLETED | no real training |
| GRPO | NOT COMPLETED | no real training |
| Post-training evaluation | NOT COMPLETED | no post-training |

Formal result mirror: `reports/real_zero_shot_baseline/summary.json`  
Smoke (distinct): `reports/real_zero_shot_baseline_smoke/`  
Config: `configs/baseline_qwen25vl_3b.yaml`  
Command pattern:

```bash
python scripts/run_baseline.py --config configs/baseline_qwen25vl_3b.yaml --run-id baseline-real-v1 --load-frames
```

---

# What the Result Means

## 33. Scientific Interpretation

```text
n = 1
accuracy = 1.0
```

**Means:** On one approved example, the untouched base model answered `right`, matching ground truth.

**Does NOT mean:** strong reasoning, causal reasoning, temporal reasoning, generalization, or post-training improvement.

Sample size matters: with one example, accuracy is either 0 or 1; it cannot estimate a population success rate or rule out lucky agreement with a short answer vocabulary (`left`/`right`).

---

# Why the Project Was Paused

## 34. Honest Ending

Paused due to **time constraints** and **dataset collection burden**:

- only one approved hidden-state gold clip
- no preference dataset
- no real DPO/GRPO
- no generalization study

This is a **scope/time decision**, not a claim that the software stack failed. The zero-shot pipeline is real and reproducible on the available gold example.

---

# What Would Have Happened Next

## 35. Research Roadmap

```text
1 approved clip
      ↓
5 approved pilot clips
      ↓
15-25 clip benchmark
      ↓
human preferences
      ↓
Bradley-Terry reward model
      ↓
DPO
      ↓
GRPO
      ↓
temporal-shuffle experiment
      ↓
reward-hacking analysis
      ↓
comparative evaluation
```

| Step | What it would test |
|------|--------------------|
| 5-clip pilot | Minimal multi-example baseline variance |
| 15-25 clip benchmark | Broader zero-shot / method comparison |
| Preferences | Human judgment of explanations |
| BT RM | Proxy scoring of responses |
| DPO | Preference-driven policy update without RM loop |
| GRPO | Objective-reward policy update via group advantages |
| Temporal shuffle | Order sensitivity diagnostic |
| Reward hacking | Proxy vs true-task divergence |
| Comparison | Which method actually helps under held-out splits |

PPO remains a **deferred** option if resources allow.

---

# HOW TO EXPLAIN THIS TO PROFESSOR XU

## 30-second version

> I’m building a VLM research prototype on magic videos where the true object state is hidden. We got a real Qwen2.5-VL-3B zero-shot pipeline working end-to-end on one human-approved Mac King clip - the model answered “right,” matching ground truth. Post-training with DPO/GRPO wasn’t started because we only have one gold example and no preference data yet; the project is paused as a reproducible prototype, not a finished learning study.

## 60-second version

> The research question is whether preference- or reward-based post-training can improve open VLMs on hidden-state and mechanism reasoning in magic/mentalism clips without shortcut learning. We implemented dataset validation, leakage checks, video preprocessing, inference, and training scaffolds. Wikimedia transparent-cup clips were rejected as gold because they leak the answer; Mac King S6 was approved. On that single example, untouched Qwen2.5-VL-3B-Instruct predicted `right` correctly. That’s n=1 - pipeline success, not evidence of general reasoning. Next would be more gold clips, preferences, then DPO for explanations and GRPO for exact-match rewards, with held-out and temporal-shuffle evaluation.

## 2-minute version

Use [`PROFESSOR_DEMO_GUIDE.md`](PROFESSOR_DEMO_GUIDE.md) screens 1-8. Emphasize: scientific question → S6 task → real correct inference → n=1 limit → intended DPO/GRPO → evaluation against shortcuts → paused status.

### "What did you actually build?"

1. Research repo architecture (`src/magic_vlm/`, configs, scripts, tests).  
2. Dataset schema, validation, leakage tooling.  
3. Video frame sampling + temporal shuffle utilities.  
4. Qwen2.5-VL load/inference on CUDA.  
5. Human review / eligibility process; S6 gold.  
6. Formal zero-shot baseline artifacts.  
7. Scaffolds for preferences, BT RM, DPO, GRPO, reward-hacking analysis.

### "What actually worked?"

Real video → preprocess → Qwen2.5-VL-3B → answer `right` = GT `right`; `n=1`, accuracy `1.0`; evidence `reports/real_zero_shot_baseline/`.

### "Why didn't you finish?"

Time + need for more approved clips and preference labels before any honest post-training claim.

### "What would you do next?"

1. Expand approved gold to ~5, then 15-25.  
2. Collect preference pairs for explanations.  
3. Run DPO + GRPO with independent held-out / temporal / hacking eval.

### "Why GRPO?"

Group-relative advantages from multiple samples per prompt; fits rule-based hidden-state reward; no separate critic (DeepSeekMath).

### "Why DPO?"

Uses preference pairs directly; no separate RM/PPO loop; natural for explanation quality.

### "Why magic?"

Hidden state + misdirection stress temporal/causal inference beyond captioning visible objects.

### "Does the one correct example prove anything?"

**No** - only that the pipeline and one zero-shot prediction matched.

### "What is the research contribution if completed?"

A hidden-state / explanation benchmark with leakage-aware splits, temporal-shuffle diagnostics, and reward-hacking analysis comparing post-training methods - **potential** contributions, not completed findings.

### Questions Professor Xu Might Ask

| Question | Concise answer |
|----------|----------------|
| Why this model? | Open, documented VLM; 3B fits consumer GPU; Instruct variant for QA. |
| Why 3B not 7B? | Memory on RTX 3060; leave headroom for future LoRA/GRPO. |
| Why not PPO? | Critic + RM + policy stack heavier; deferred. |
| Why not only SFT? | Research target is preference/reward post-training & shortcut analysis. |
| Why reward hacking? | Proxy rewards can rise without true task skill. |
| Why held-out tricks? | Prevent memorizing performer/setup. |
| Why video not images? | Hidden state depends on event order. |
| Why doesn’t one correct answer prove reasoning? | n=1; no variance; possible shortcuts. |
| Why S6 accepted but Wikimedia rejected? | S6 no-reveal + human APPROVE; Wikimedia leakage/visibility. |
| What next experiment? | Approve more gold clips, then zero-shot on the 5-clip pilot before any training. |

Diagram: [`assets/professor_talking_points.svg`](assets/professor_talking_points.svg).

---

# References (selected)

- Qwen2.5-VL-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct  
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300  
- DPO (Rafailov et al.): https://arxiv.org/abs/2305.18290  
- InstructGPT / RLHF (Ouyang et al.): https://arxiv.org/abs/2203.02155  
- Hugging Face TRL docs: https://huggingface.co/docs/trl  
- Cui et al. 2011 Mac King stimuli: https://pmc.ncbi.nlm.nih.gov/articles/PMC3202226/  
- Project plan: [`../magic-vlm-research-plan-v2.md`](../magic-vlm-research-plan-v2.md)

---

*End of complete guide. For the live operational banner, regenerate `PROJECT_STATUS.md` via `python scripts/project_health.py`.*
