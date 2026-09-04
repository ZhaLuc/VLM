# Visual overview

Short cues for each diagram. For full detail, see [`TECHNICAL_GUIDE.md`](TECHNICAL_GUIDE.md).

---

## 1. Idea

![idea](assets/01_idea.svg)

- magic video + question
- VLM answer
- score / optional later training
- prototype paused after first real test

---

## 2. Pipeline

![backbone](assets/02_backbone.svg)

- validate examples
- sample frames
- run VLM
- compare to ground truth
- write artifacts

---

## 3. Task

![task](assets/03_task.svg)

- Mac King S6
- fake toss, no reveal
- hidden coin location
- question: which hand?
- ground truth: right

---

## 4. Work completed

![process](assets/04_process.svg)

- repository + tests
- rejected leaky clips
- approved S6 only
- CUDA + Qwen2.5-VL-3B
- real zero-shot baseline
- DPO / GRPO not trained

---

## 5. Result

![result](assets/05_result.svg)

- model: right
- ground truth: right
- n = 1
- pipeline verified
- not a reasoning claim

---

## 6. Planned learning

![learning](assets/06_learning.svg)

- DPO: preference pairs / explanations
- GRPO: group-relative rewards / hidden state
- training code scaffolded
- no real post-training run

---

## 7. Next steps

![next](assets/07_next.svg)

- expand gold set
- collect preferences
- DPO and/or GRPO
- held-out + anti-shortcut evaluation

---

## Key paths

- gold: `data/examples/hidden_state_pilot.jsonl`
- result: `reports/real_zero_shot_baseline/`
- config: `configs/baseline_qwen25vl_3b.yaml`
- status: `PROJECT_STATUS.md`
