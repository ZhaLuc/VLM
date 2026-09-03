# Visual Reasoning and Explanation from Magic and Mentalism Demonstrations

A **paused** VLM research prototype studying visual reasoning and explanation from
magic/mentalism demonstration clips — documented as a teaching archive, not a
completed post-training study.

**Working title:** Visual Reasoning and Explanation from Magic and Mentalism
Demonstrations: A Post-Training Study of Vision-Language Models

---

## What this project is

A research codebase for evaluating (and, if resumed, post-training) open
vision-language models on **hidden-state** and mechanism questions in short
magic/mentalism videos — with leakage checks, temporal diagnostics, and
preference/reward scaffolds (DPO / GRPO).

## Current result

Real `Qwen/Qwen2.5-VL-3B-Instruct` zero-shot inference on one human-approved
Mac King S6 example (`Movie6.MP4`) answered **`right`**, matching ground truth.
Formal evidence: `reports/real_zero_shot_baseline/` (`n = 1`, accuracy `1.0`).

That is **pipeline success on one example**. It does **not** prove strong
reasoning, causal/temporal understanding, generalization, or post-training gains.

## Current limitation

The post-training study was **paused** before DPO/GRPO on the real VLM because
the gold set was not yet large enough (1 of 5 pilot clips approved; S7 pending)
and time became limited. Preference labels were not collected.

**Banner:** `PAUSED - ZERO-SHOT PROTOTYPE COMPLETE` (see `PROJECT_STATUS.md`).

---

## START HERE

1. **[Project Overview (visual homepage)](docs/index.html)** — open locally in a browser
2. **[Professor Xu 2-Minute Demo](docs/PROFESSOR_DEMO_GUIDE.md)**
3. **[Complete Technical Guide](docs/PROJECT_COMPLETE_GUIDE.md)**
4. **[Project Status](PROJECT_STATUS.md)**
5. **[Repository Architecture](docs/ARCHITECTURE.md)**

Also useful before a meeting:

- [One-page Professor Xu sheet](docs/PROFESSOR_XU_ONE_PAGE.md)
- [CUDA / Qwen environment notes](docs/CUDA_ENVIRONMENT.md)
- [Formal S6 baseline evidence](reports/real_zero_shot_baseline/)

### Open the visual homepage locally

```bash
# from the repo root (Windows PowerShell example)
start docs/index.html
# or: python -m http.server 8000 --directory docs
# then visit http://localhost:8000/
```

---

## Honest scientific snapshot

| Done | Not done |
|------|----------|
| Repo architecture, tests, health audits | 5-clip / 15–25 clip benchmarks |
| Dataset schema, validation, leakage tooling | Human preference dataset |
| Video preprocessing + frame sampling | Bradley-Terry RM on real prefs |
| CUDA + Qwen2.5-VL-3B load | Real DPO / GRPO / PPO training |
| Human-approved S6 gold | Temporal/causal post-training study |
| Real zero-shot baseline (`n=1`, correct) | Evidence of post-training improvement |

Wikimedia / PeerJ cups-and-balls clips remain **controls**
(`NOT_SUITABLE_FOR_HIDDEN_STATE`), not gold. Gold manifest:
`data/examples/hidden_state_pilot.jsonl`. Candidate review:
`reports/hidden_state_candidates/index.html`.

---

## Layout

```
configs/               # experiment YAML
data/examples/         # manifests / reviews (raw videos not committed)
docs/                  # teaching archive + technical docs + assets/
reports/               # committed health + baseline evidence mirrors
scripts/               # thin CLI wrappers
src/magic_vlm/         # package
tests/                 # unit + interface smoke tests
PROJECT_STATUS.md      # live regenerated status
```

Diagrams: `docs/assets/*.svg`.

---

## Setup

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[dev,video,models]"
```

CUDA notes for the real baseline: `docs/CUDA_ENVIRONMENT.md`.

## Reproducibility init (no model load)

```bash
python -m pytest
magic-vlm-init --config configs/baseline_stub.yaml
```

## Formal zero-shot baseline (already recorded)

```bash
magic-vlm-baseline --config configs/baseline_qwen25vl_3b.yaml --run-id baseline-real-v1 --load-frames
```

Committed mirror: `reports/real_zero_shot_baseline/`. Smoke (distinct):
`reports/real_zero_shot_baseline_smoke/`. See `docs/BASELINE.md`.

## Project health

```bash
magic-vlm-project-health
# or: python scripts/project_health.py
```

## Training scaffolds (not run as completed research)

Preference schema, BT reward-model, DPO, GRPO, temporal shuffle, and
reward-hacking tools exist as modules/scripts with toy/fixture paths. See
`docs/DPO.md`, `docs/GRPO.md`, `docs/REWARD_MODEL.md`, `docs/TEMPORAL_SHUFFLE.md`,
`docs/REWARD_HACKING.md`. **Do not treat scaffold smoke as post-training results.**

Research plan notes: `magic-vlm-research-plan-v2.md`.

## License

MIT (see `pyproject.toml`). Research plan documents remain project notes.
