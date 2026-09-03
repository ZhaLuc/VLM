# Walkthrough

Scroll. Talk from the bullets.

Prep notes (separate): [`STUDY_GUIDE.md`](STUDY_GUIDE.md)

---

## 1

![idea](assets/01_idea.svg)

- magic video + question
- VLM answers
- score / maybe train later
- paused after first real test

---

## 2

![backbone](assets/02_backbone.svg)

- validated examples
- sample frames
- ask model
- compare to answer key
- save run

---

## 3

![task](assets/03_task.svg)

- Mac King S6
- fake toss, no reveal
- hidden coin
- question: which hand?
- ground truth: right

---

## 4

![process](assets/04_process.svg)

- built repo + tests
- rejected leaky clips
- approved S6 only
- CUDA + Qwen 3B
- real zero-shot run
- stopped before DPO/GRPO

---

## 5

![result](assets/05_result.svg)

- model: right
- key: right
- n = 1
- pipeline works
- not a reasoning proof

---

## 6

![learning](assets/06_learning.svg)

- DPO: preference pairs / explanations
- GRPO: group scores / hidden-state reward
- code scaffolded
- no real training yet

---

## 7

![next](assets/07_next.svg)

- more gold clips
- preference labels
- DPO and/or GRPO
- held-out + anti-shortcut checks

---

## Repo pointers

- gold: `data/examples/hidden_state_pilot.jsonl`
- result: `reports/real_zero_shot_baseline/`
- config: `configs/baseline_qwen25vl_3b.yaml`
- status: `PROJECT_STATUS.md`
