# Full Study Guide (for you, before the meeting)

This document is **for your own preparation**. It is not the meeting slides.

For the meeting itself, open [`WALKTHROUGH.md`](WALKTHROUGH.md) and scroll the diagrams while you talk.

Here, the goal is different: after reading this, you should be able to explain **what the project is**, **what you built**, **what actually worked**, **what did not happen**, and **what every important technical word means** - in plain language, but with enough depth that follow-up questions do not surprise you.

**Honesty first:** this is a **paused research prototype**. The real completed experiment is a **zero-shot** run on **one** approved example (`n = 1`). Post-training was prepared in code but **not** completed on real preference/reward data.

---

## How to use this guide

1. Read sections 1-4 once for the big picture.
2. Read section 5 (glossary) and keep it open while you study later sections.
3. Read sections 6-11 carefully - that is "what the system actually does."
4. Read sections 12-16 - that is "what learning would have meant."
5. Read section 17 - that is your result and how to talk about it without overclaiming.
6. Drill section 18 (Q&A) out loud.
7. The night before: skim [`WALKTHROUGH.md`](WALKTHROUGH.md), then re-read section 17 and 18 here.

---

## 1. The project in one honest paragraph

You built a research codebase that can take a short magic video, sample frames, ask an open vision-language model a question about a **hidden state** (for example, which hand still holds a coin after an apparent transfer), record the model's raw text answer, and score it against a human-approved answer key.

You successfully ran that full path on real hardware with a real model (**Qwen2.5-VL-3B-Instruct**) on one human-approved Mac King clip (**S6**). The model answered `right`, which matched the ground truth. That proves the **pipeline** works.

You did **not** finish the intended scientific study: collecting a larger gold set, collecting human preferences, training with DPO or GRPO, and measuring whether post-training actually improves hidden-state reasoning without shortcut learning.

---

## 2. The research question (why this is science, not a demo app)

The intended question was:

> Can preference-based and/or reward-based post-training improve an open VLM's ability to infer hidden states and explain mechanisms in magic/mentalism videos, while generalizing beyond a small training set rather than exploiting shortcuts?

Break that into pieces:

| Phrase | Meaning |
|--------|---------|
| open VLM | A model whose weights you can load yourself (here, Qwen), not a closed API black box |
| hidden states | The true situation is not fully visible at the end (coin still in a hand, but not shown) |
| explain mechanisms | Not only "left/right," but eventually "what method happened" |
| post-training | Extra training **after** the public pretrained checkpoint |
| preference-based | Humans say answer A is better than B |
| reward-based | A scoring function gives a number for each answer |
| generalizing | Works on unseen tricks / performers / cameras / wordings |
| shortcuts | Looking transparent, memorizing one performer, gaming the reward, etc. |

This is scientific because the hard part is not "make a chatbot that watches video." The hard part is proving that any improvement is **real capability**, not leakage or reward hacking.

---

## 3. What this project is not

Say these out loud so you do not accidentally mis-describe it:

- Not robotics
- Not physically performing magic
- Not a magic game
- Not "we trained a model and it learned causal reasoning"
- Not "accuracy 1.0 means the model is strong"
- Not a finished RLHF paper result

It **is**: a reproducible evaluation spine + one real zero-shot success + unfinished training scaffolds.

---

## 4. The mental model: three layers

Keep this picture in your head:

```text
Layer A - Research idea
  hidden-state / explanation questions on magic clips
  + planned DPO / GRPO comparison
  + anti-shortcut evaluation

Layer B - Engineering spine (mostly done for n=1)
  data -> validate -> sample frames -> VLM -> parse -> score -> save artifacts

Layer C - Learning methods (code exists; real study not done)
  preferences, reward model, DPO, GRPO, temporal shuffle diagnostics
```

When someone asks "what did you build?", answer with **Layer B**.  
When someone asks "what was the research goal?", answer with **Layer A**.  
When someone asks "did training help?", answer: **Layer C was not completed.**

---

## 5. Glossary (learn these words cold)

### Models and multimodal basics

**LLM (large language model)**  
A neural network trained mainly on text to predict the next token.

**Vision model**  
A model that turns images into numerical features.

**Multimodal model**  
A model that combines more than one kind of input (here: vision + language).

**VLM (vision-language model)**  
A multimodal model that takes images/video frames and text, and usually outputs text. Your model is a VLM.

**Checkpoint / weights / parameters**  
The saved numbers that define the model's behavior. "Loading Qwen" means loading those numbers into memory.

**Base / Instruct**  
"Base" often means pretrained for next-token prediction. "Instruct" means further tuned to follow instructions / answer questions. You used **Qwen2.5-VL-3B-Instruct**.

**Zero-shot**  
Ask the model to do the task **without** task-specific fine-tuning for your magic dataset. Your real run was zero-shot.

**Fine-tuning / post-training**  
Updating the checkpoint further on your task data or preference/reward signals.

**LoRA / PEFT**  
Parameter-efficient fine-tuning: train small adapter weights instead of all model weights. Your DPO/GRPO scaffolds are built around this idea for memory reasons. You did not complete real VLM post-training.

### Tokens and generation

**Token**  
A chunk of text the model reads/writes (often a subword). An answer like `right` is one or a few tokens.

**Logits**  
Raw scores the model assigns to each possible next token.

**Probability**  
Normalized chance of each next token (often via softmax).

**Generation / decoding**  
Repeatedly choosing next tokens until the answer is complete.

**Greedy decoding**  
Always pick the highest-probability next token. Your formal baseline used this (`temperature: 0.0`, `do_sample: false`).

**Temperature / sampling**  
Higher temperature / sampling makes answers more random. Your baseline avoided that for reproducibility.

**Raw text vs parsed answer**  
Raw text is exactly what the model emitted. Parsing extracts a comparable label (for example `right`). Your pipeline stores both so you can audit mistakes.

### Data and evaluation

**Manifest / JSONL**  
A text file where each line is one JSON example. Your gold set is `data/examples/hidden_state_pilot.jsonl`.

**Ground truth / answer key**  
The human-approved correct label for scoring.

**Split (`train` / `val` / `held_out`)**  
Which examples are for learning, tuning, or final untouched evaluation. Your S6 gold row is `held_out`.

**Leakage**  
Train and test are not truly separate (same clip, same trick identity, answer visible in the video, etc.).

**Gold example**  
An example approved for real scoring. Only S6 is gold right now.

**Control example**  
Useful for comparison or pipeline testing, but **not** counted as hidden-state gold (for example transparent-cup Wikimedia clips).

**Exact match**  
Score 1 if normalized prediction equals ground truth, else 0.

**n**  
Number of evaluated examples. Yours is **n = 1**.

**Accuracy**  
Correct / total. With n = 1, accuracy is either 0 or 1. That is why it cannot prove general ability.

### Learning and RL terms

**Loss**  
A number that says how wrong the current prediction is (or how bad the preference/reward objective is). Training tries to reduce the badness of that objective.

**Gradient**  
Direction to nudge parameters to improve the objective.

**SFT (supervised fine-tuning)**  
Show desired answers and increase their probability. Optional idea here; not your main completed experiment.

**Preference pair**  
Two answers for the same prompt; a human marks which is better.

**Reward**  
A scalar score for an answer. Can be rule-based (exact match) or learned.

**Reward model (RM)**  
A model trained to predict how much humans would like an answer.

**Bradley-Terry**  
A classic model for preferences: if humans prefer A over B, the RM should assign a higher score to A than B. Intuition:  
`P(A beats B) = sigmoid(score_A - score_B)`.

**RL (reinforcement learning)**  
Learn from rewards after actions. In language settings, the "action" is generating text.

**Policy**  
The model that generates answers (your VLM).

**RLHF**  
Reinforcement Learning from Human Feedback: preferences -> reward model -> RL update of the policy.

**PPO**  
A common RL algorithm with careful update sizes; usually needs extra components (often a critic). Deferred in this project.

**DPO**  
Direct Preference Optimization: learn from preference pairs without a separate reward-model + PPO loop.

**GRPO**  
Group Relative Policy Optimization: sample several answers for one prompt, score them, update using relative advantages inside that group (no separate critic). Attractive for rule-based rewards.

**Advantage**  
How much better/worse an answer is than a baseline expectation. In GRPO, that baseline is the group mean (scaled by group std).

**KL regularization**  
A penalty that keeps the updated model from drifting too far from a reference model.

**Reward hacking**  
The model improves the proxy score without truly getting better at the intended task.

**Temporal shuffle**  
Show the same frames in scrambled order. Used as a diagnostic: if order mattered, performance should often change.

---

## 6. Why magic and mentalism?

Magic demonstrations are not chosen because the project is "about entertainment." They are chosen because they force several hard multimodal skills at once:

1. **Hidden state:** the true object location can be occluded.
2. **Misdirection:** gaze and gesture push attention toward the wrong place.
3. **Temporal structure:** earlier frames matter; the last frame can lie.
4. **Apparent vs true cause:** the visible "toss" may not be the real transfer.
5. **Explanation:** beyond a one-word label, one may ask what method occurred.

That is why transparent cups and late reveals were rejected as gold: if the answer is already visible, you are no longer testing hidden-state inference.

Source material you actually used includes Mac King supplementary videos from Cui et al. 2011 (Frontiers in Human Neuroscience), locally `Movie1.MP4` ... `Movie7.MP4`. PeerJ / Wikimedia cups-and-balls clips were inspected and kept as controls.

---

## 7. The concrete task: Mac King S6

### What happens in the clip

At a high level:

1. Coin is visible in the **right** hand.
2. Performer does an **apparent** transfer / fake toss toward the left.
3. The coin is retained (hidden) on the right.
4. The evaluation clip **ends without the reveal** that would show an empty left hand (that reveal exists in S1, the counterpart).
5. Question: which hand contains the coin after the apparent transfer?
6. Answer key: **right**

### Why this is not "classify the last frame"

If you only looked at the ending pose / gaze, you might guess left. The correct answer depends on understanding the earlier fake transfer. That is why video (ordered frames) matters more than a single still.

### Related clips (so you can discuss the set)

| Clip | Role |
|------|------|
| S1 | Same trick family with reveal - control / revealed counterpart of S6 |
| S2 | Real toss with reveal - control |
| S3/S4/S5 | Other study conditions; mostly controls / not gold |
| **S6** | No-reveal fake toss - **only approved gold** |
| S7 | No-reveal real toss candidate - still **PENDING** |

You did not lower standards just to get five gold clips. That is a feature, not a bug.

---

## 8. Dataset structure (what an "example" is)

Each scored example is basically:

- identity: `example_id`, `clip_id`, `trick_id`, `performer_id`, `camera_id`
- media: video path + content hash + duration/fps metadata
- task fields: `question`, `ground_truth`, task type (`hidden_state`)
- split: usually `held_out` for final eval
- provenance: where it came from, license notes, who approved it

Why so much metadata? Because without trick/performer/camera tags, you cannot detect leakage or claim generalization.

Gold file:

- `data/examples/hidden_state_pilot.jsonl` (currently one row: S6)

Human decisions / reviews live under `data/examples/` (for example `human_review_decisions.json`, `mac_king_review.jsonl`).

---

## 9. Preprocessing: what happens to the video

The model does not "watch" the MP4 as a continuous movie in your pipeline the way a human does. Your project:

1. Opens the video.
2. Chooses a small set of frame indices.
3. Decodes those frames.
4. Sends those frames + the text question into the VLM.

### Uniform frame sampling

For the formal S6 run:

- `max_frames: 8`
- `sample_strategy: uniform`
- Actual indices used: `0, 31, 61, 92, 123, 154, 184, 215`
- Source fps about 29.97
- About 216 frames total in Movie6
- `temporal_shuffled: false`

"Uniform" means spread frames across the clip instead of taking only the beginning or only the end.

### Why this matters

- Too few frames: miss the critical moment.
- Only last frames: miss the fake toss evidence.
- Shuffled frames: same pixels, broken order - useful later as a diagnostic, not used in the formal baseline.

### Content hash

The run recorded a SHA-256 of the file (`de8e1768...`). That helps prove which bytes were evaluated.

Code center of gravity: `src/magic_vlm/video.py`.

---

## 10. Model loading and inference

### What model, what hardware

- Model id: `Qwen/Qwen2.5-VL-3B-Instruct`
- Checkpoint label in the run: `untouched_base_3b` (not a DPO/GRPO-trained adapter)
- Device: `cuda:0`, NVIDIA GeForce RTX 3060
- Torch: CUDA build (`2.13.0+cu130` in the recorded environment)
- Why 3B not 7B: fits consumer GPU memory better and leaves headroom for future LoRA training ideas

### What inference means here

1. Build a prompt from the example (question + instruction template).
2. Attach sampled frames.
3. Generate text with the configured decoding settings.
4. Keep **raw** output.
5. Parse a label for scoring.

Formal generation settings:

- `max_new_tokens: 128`
- `temperature: 0.0`
- `do_sample: false` (greedy)
- latency on the recorded run: about 2.6 seconds for that one example

Code: `src/magic_vlm/models.py`, `inference.py`, `baseline.py`.

Config: `configs/baseline_qwen25vl_3b.yaml`.

Command shape:

```bash
python scripts/run_baseline.py --config configs/baseline_qwen25vl_3b.yaml --run-id baseline-real-v1 --load-frames
```

---

## 11. Evaluation: how "correct" is decided

For hidden-state labels, the main metric is **exact match** after light normalization/parsing.

For S6:

- raw text: `right`
- parsed answer: `right`
- ground truth: `right`
- correct: true
- parse_failed: false

Aggregate metrics in the report:

- `n_examples = 1`
- `n_correct = 1`
- `n_incorrect = 0`
- `overall_accuracy = 1.0`

Important nuance:

- Exact match is clear and strict.
- It does **not** measure explanation quality.
- It can be gamed later if a training reward equals the same metric and the model finds shallow tricks.

Evidence folders:

- Formal: `reports/real_zero_shot_baseline/` (label `REAL_ZERO_SHOT_BASELINE`)
- Earlier smoke (distinct): `reports/real_zero_shot_baseline_smoke/`
- Full local run dir (gitignored): `runs/baseline-real-v1/`

---

## 12. What you actually did (process timeline)

Use this as your "process" story.

### A. Define the research scope

You framed a post-training study on hidden-state / explanation reasoning with anti-shortcut evaluation, not a generic captioning app.

### B. Build repository architecture

Package under `src/magic_vlm/`, configs, scripts, tests, experiment metadata, project health audits.

### C. Implement data integrity machinery

Schema for examples, validation, leakage checks, eligibility thinking for hidden-state gold vs controls.

### D. Implement video tooling

Probe videos, sample frames, optional temporal shuffle utilities.

### E. Implement inference + baseline runner

Load stub or real Qwen, run batches, preserve raw outputs, write immutable-style baseline artifacts.

### F. Review real footage scientifically

- Wikimedia / PeerJ cups-and-balls: rejected as hidden-state gold (visibility / reveal leakage), kept as controls.
- Mac King set inspected.
- Human APPROVE for S6 only.
- S7 left pending on purpose.

### G. Make the real stack work

CUDA PyTorch, model weights in HF cache, real MP4 decode, real generation on GPU.

### H. Record the formal zero-shot baseline

`baseline-real-v1` / `REAL_ZERO_SHOT_BASELINE`, n = 1, correct.

### I. Pause

Not because the code collapsed. Because the dataset was too small and time ran out for an honest learning study.

---

## 13. Learning methods, explained slowly

### 13.1 Pretraining (already done by others)

Before you touched anything, Qwen was trained on huge multimodal data to predict tokens and follow instructions. You start from that public Instruct checkpoint.

### 13.2 SFT (optional, not your main completed path)

Show: video + question + desired answer.  
Update the model so that answer becomes more likely.  
Simple, but it only imitates demonstrations. It does not directly use "A is better than B" preferences.

### 13.3 Human preferences

Example:

- Answer A: "The coin remains in the right hand."
- Answer B: "It moved to the left hand."
- Human: A is better.

That pair is supervision for explanation quality. **You did not collect a real preference dataset.**

### 13.4 Reward models and Bradley-Terry

A reward model maps (video, question, answer) -> score.  
Bradley-Terry training pushes preferred answers to higher scores than rejected ones.

Why useful: RL algorithms want a number.  
Why dangerous: the number is a proxy and can be wrong.

### 13.5 RLHF sketch

```text
base model -> many candidate answers -> human preferences
  -> reward model -> RL update -> new model
```

### 13.6 PPO (deferred)

PPO updates the policy using reward/advantage signals while clipping updates so the model does not change too violently. It often needs:

- policy (VLM)
- reference model
- reward model
- value/critic model

That is memory-heavy on a 3B VLM + 12GB GPU, so PPO was deferred.

### 13.7 DPO (planned for explanations; not completed)

DPO uses preference pairs to update the policy directly.  
No separate RM + PPO loop.  
Great conceptual fit for "which explanation is better?"  
Blocked in practice by missing preference data.

### 13.8 GRPO (planned for hidden-state objective reward; not completed)

For one prompt, sample multiple answers, score them (for example exact-match style reward), compute group-relative advantages, update the policy.

Numeric intuition from the research plan:

- rewards: 0.9, 0.7, 0.2, 0.1
- mean: 0.475
- population std: about 0.3345
- advantages: about +1.27, +0.67, -0.82, -1.12

Better-than-average answers get positive advantage; worse ones get negative. Relative comparison inside the group is the key idea. Associated with DeepSeekMath.

### 13.9 Method comparison cheat sheet

| Method | Learns from | Needs separate RM? | Online sampling? | Separate critic? | Your intended use | Status |
|--------|-------------|--------------------|------------------|------------------|-------------------|--------|
| SFT | gold answers | no | no | no | optional | not main result |
| DPO | preference pairs | no | no | no | explanations | scaffold only |
| PPO | rewards | usually | yes | yes | deferred | not run |
| GRPO | rewards (group-relative) | optional | yes | no | hidden-state | scaffold only |

---

## 14. Temporal shuffle and causal language (careful)

### Temporal shuffle

Same frames, different order. If accuracy collapses, the model was sensitive to order (or at least to that presentation). If nothing changes, maybe it ignored order or used non-temporal shortcuts.

Your formal baseline did **not** shuffle. Shuffle tooling exists for later diagnostics.

### Causal reasoning

Do not casually claim your project "tests causation" as a completed result. Magic makes causal labels hard because apparent cause and true mechanism disagree by design. The repo has hooks for richer temporal/causal annotations, but that study was not completed.

Safe sentence:

> "We care about temporal structure and mechanism inference. We have diagnostics planned for order sensitivity. We did not complete a causal post-training evaluation."

---

## 15. Reward hacking (why independent eval matters)

Imagine:

- proxy reward: 0.40 -> 0.90
- true task accuracy: 50% -> 51%

The model learned the scorer, not the skill.

Common shortcut families:

- verbosity / confidence fluff
- keyword stuffing
- answer-distribution bias
- parser exploitation
- visual leakage (transparent cups, reveals)

This is why gold standards, held-out splits, and metrics that are not identical to the training reward all matter.

---

## 16. What exists in code vs what was completed

### Completed for real

- repo architecture + extensive tests
- dataset schema / validation / leakage tooling
- video preprocessing / frame sampling
- CUDA + Qwen load
- human-approved S6 gold
- real zero-shot baseline (n = 1, correct)
- project health banner: paused prototype complete

### Scaffolded / toy-tested, not completed as real research

- preference annotation workflow
- Bradley-Terry reward model training on real human prefs
- real DPO on Qwen with real preferences
- real GRPO on Qwen with real objective rewards
- temporal-shuffle post-training study
- reward-hacking before/after analysis on real post-training
- 5-clip pilot and 15-25 clip benchmark

If asked "so you implemented DPO?", answer precisely:

> "I implemented the training path and tested it on smoke/toy setups. I did not run DPO on the real VLM with real magic preference data."

---

## 17. The result: how to interpret n = 1 without lying

### What is true

- Real video file.
- Real model weights.
- Real GPU inference.
- Human-approved question and ground truth.
- Model output `right` matched ground truth.
- End-to-end research pipeline works.

### What is not true

- "The model has strong hidden-state reasoning."
- "We demonstrated causal reasoning."
- "Post-training improved the model."
- "Accuracy 1.0 means it generalizes."

### Why sample size matters

With one example, you cannot estimate a stable success rate. You cannot separate luck, short-answer priors (`left`/`right`), or genuine multi-frame inference. You need more approved clips before any learning claim.

### Best one-sentence interpretation

> The repository successfully demonstrated a real zero-shot multimodal inference pipeline on one human-approved hidden-state example, but the planned post-training research experiment was not completed.

Memorize that sentence.

---

## 18. Professor Q&A drill (practice out loud)

**Q: What did you build?**  
A: A gated evaluation pipeline for hidden-state VLM questions on magic clips, plus unfinished DPO/GRPO scaffolds.

**Q: What actually worked?**  
A: Qwen2.5-VL-3B on Mac King S6 answered `right` correctly; n = 1.

**Q: Does that prove reasoning?**  
A: No. Pipeline success on one example is not a generalization or reasoning proof.

**Q: Why magic?**  
A: Hidden state + misdirection stress temporal multimodal inference beyond captioning visible objects.

**Q: Why reject Wikimedia clips?**  
A: Transparent cups / visible state / reveals leak the answer. Keeping them as controls preserved the benchmark standard.

**Q: Why S6 not S7?**  
A: S6 was human-approved. S7 still had unresolved review uncertainty, so it stayed pending.

**Q: Why this model / why 3B?**  
A: Open Instruct VLM; 3B fits RTX 3060 and leaves room for future adapters.

**Q: Why not only SFT?**  
A: The research target was preference/reward post-training and shortcut analysis, not only imitation.

**Q: Why DPO?**  
A: Fits explanation preferences; no separate RM/PPO loop.

**Q: Why GRPO?**  
A: Fits rule-based hidden-state rewards via group-relative advantages without a critic.

**Q: Why not PPO first?**  
A: Heavier stack (often policy + reference + reward + critic); deferred for complexity/memory.

**Q: Why pause?**  
A: Time + only one gold clip + no preference dataset. Software spine worked; learning study was not yet honest to run.

**Q: What next?**  
A: More gold clips, preference collection, then DPO and/or GRPO with held-out and anti-shortcut evaluation.

**Q: Where is the evidence?**  
A: `reports/real_zero_shot_baseline/`, config `configs/baseline_qwen25vl_3b.yaml`, gold `data/examples/hidden_state_pilot.jsonl`.

---

## 19. File map (so you can point in the repo)

| Need | Path |
|------|------|
| Meeting talk track + images | `docs/WALKTHROUGH.md` |
| This deep study guide | `docs/STUDY_GUIDE.md` |
| Status banner | `PROJECT_STATUS.md` |
| Gold example | `data/examples/hidden_state_pilot.jsonl` |
| Formal result | `reports/real_zero_shot_baseline/` |
| Baseline config | `configs/baseline_qwen25vl_3b.yaml` |
| Package code | `src/magic_vlm/` |
| Tests | `tests/` (230 passed at last cleanup) |
| Original long research plan notes | `magic-vlm-research-plan-v2.md` (design intent; not completed results) |

### Important modules

| Module | Job |
|--------|-----|
| `schemas.py` / `dataset.py` | example records |
| `validate.py` | integrity + leakage checks |
| `video.py` | frame sampling / shuffle |
| `models.py` | load stub or Qwen |
| `inference.py` | generate + parse |
| `evaluation.py` / `baseline.py` | score + baseline runner |
| `preferences.py` / `annotation.py` | preference I/O / collection tools |
| `reward_model.py` / `rewards.py` | RM + objective rewards |
| `dpo.py` / `grpo.py` | training scaffolds |
| `temporal.py` | ordered vs shuffled diagnostic runner pieces |
| `reward_hacking.py` | divergence diagnostics |
| `project_health.py` | live status audit |

---

## 20. A study checklist for the day before

- [ ] I can state the research question in one breath.
- [ ] I can explain VLM vs LLM without notes.
- [ ] I can walk S6: visible right -> fake toss -> hidden -> question -> `right`.
- [ ] I can explain preprocessing: 8 uniform frames, no shuffle in the formal run.
- [ ] I can explain zero-shot greedy decoding.
- [ ] I can say what n = 1 does and does not mean.
- [ ] I can explain DPO vs GRPO in one sentence each.
- [ ] I can explain why Wikimedia clips were rejected.
- [ ] I can explain why the project paused without sounding defensive.
- [ ] I can point to the evidence folder.
- [ ] I can open `WALKTHROUGH.md` and talk while scrolling.

---

## 21. Final rehearsal script (2 minutes)

> I'm studying whether post-training can improve an open VLM on hidden-state reasoning in magic videos without shortcut learning. I built a full evaluation spine: validated examples, frame sampling, Qwen2.5-VL inference, exact-match scoring, and artifact logging. I rejected leaky Wikimedia clips as gold and human-approved one Mac King no-reveal clip, S6. On that clip, untouched Qwen2.5-VL-3B answered "right," matching ground truth. That is n equals one: it shows the pipeline is real, not that the model has general reasoning or that training helped. DPO and GRPO code is scaffolded, but I paused before real post-training because I still needed more gold clips and preference labels. Next step would be expand the gold set, collect preferences, then train and evaluate under held-out and anti-shortcut checks.

If you can say that cleanly, you are ready for the meeting. Use [`WALKTHROUGH.md`](WALKTHROUGH.md) as the visual companion while you speak.
