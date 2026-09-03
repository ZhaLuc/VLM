# Project health audit

Independent acceptance check of what actually works vs what is only scaffolded.

```bash
magic-vlm-project-health
python scripts/project_health.py
python scripts/project_health.py --skip-tests
python scripts/project_health.py --skip-stub-baseline
```

## Outputs

| Path | Purpose |
|------|---------|
| `PROJECT_STATUS.md` | Human-readable status |
| `reports/project_status.html` | Visual dashboard (open in a browser) |
| `reports/project_health/audit.json` | Machine-readable source of truth |
| `reports/project_health/project_status.html` | Same dashboard |

## Integrity

- `PASS` requires meaningful evidence (not just imports).
- Missing videos / GPU / weights stay `BLOCKED`.
- Synthetic/stub success is never reported as a real research baseline.
- Hidden-state dataset counts (`WIKIMEDIA CONTROLS`, `MAC KING CANDIDATES`,
  `HIDDEN-STATE GOLD`) are read from
  `data/examples/hidden_state_candidate_inventory.json`. They are not
  hard-coded.
