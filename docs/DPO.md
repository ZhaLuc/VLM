# DPO post-training (explanation preferences)

Compares an **untouched base checkpoint** to a **DPO-adapted** checkpoint.
Never overwrites immutable baseline runs.

## Verified local stack (probe)

Run:

```bash
python -c "from magic_vlm.dpo import probe_dpo_stack; import json; print(json.dumps(probe_dpo_stack().to_dict(), indent=2))"
```

Documented expectations for this repo host (as of implementation):

| Component | Role |
|-----------|------|
| Transformers | Model/processor load |
| TRL `DPOTrainer` | DPO loss + training loop |
| PEFT LoRA | Parameter-efficient adaptation |
| CUDA | Required for practical Qwen2.5-VL DPO |

TRL vision notes:

- VLM DPO expects `image` / `images` columns and `max_length=None` to avoid truncating image tokens.
- Qwen2.5-VL needs `mm_token_type_ids` forwarding (TRL issue #5277 / PR #5279). The probe reads the installed `dpo_trainer.py` for this symbol.

If `ready_for_vlm_dpo` is false, **do not invent API workarounds**. Use text smoke for plumbing only.

## Integrity

- Do not equate DPO loss decrease with reasoning improvement.
- Do not claim success because rejected-response likelihood falls.
- Do not train on `held_out` preferences.
- Do not select checkpoints using final held-out test scores (`checkpoint_selection: last_train_step`).
- Preserve raw preference texts; adapter does not rewrite candidates.

## Commands

```bash
# Stack probe
python -c "from magic_vlm.dpo import probe_dpo_stack; print(probe_dpo_stack())"

# Text plumbing smoke (creates a tiny local LM; no Hub / no VLM)
magic-vlm-train-dpo --config configs/dpo_smoke_text.yaml --smoke-local-lm

# Score/compare plan only (no test-set cherry-picking)
# After a real run, use a *new* eval run_id with the established baseline protocol.
```

## Outputs

`runs/dpo/<run_id>/`:

- `checkpoint/` — DPO (LoRA) weights; isolated from baseline
- `train_metadata.json` — beta, PEFT, seed, hardware, splits
- `dpo_train_records.jsonl` / `dpo_val_records.jsonl`
- `stack_probe.json` — verified compatibility
- `DISCLAIMER.json` / `comparison_plan.json`
- `BASELINE` protection checks when `baseline_run_dir` is set
