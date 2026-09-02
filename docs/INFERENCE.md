# VLM inference interface

Inference-only path for Qwen2.5-VL (and stub models). **No post-training.**

## Verified APIs (Transformers 5.16.1, this machine)

- Class: `transformers.Qwen2_5_VLForConditionalGeneration`
- Processor: `AutoProcessor` / `Qwen2_5_VLProcessor.apply_chat_template`
- Documented video message: `{"type": "video", "path": "..."}` with optional `fps`
- `model.generate(..., max_new_tokens=, do_sample=, temperature=, top_p=, top_k=)`
- New-token decoding: trim `output_ids` by `input_ids` length, then `batch_decode`

This project **defaults to passing already sampled frames** (`video_input_mode=project_sampled_frames`) so temporal-shuffle experiments can reuse the same `ordered_indices`. That is a research choice; it is not identical to letting the processor sample a raw file at `fps=1`.

## What is not claimed

- Bitwise-deterministic GPU decoding
- That 7B video inference was executed on this host (see hardware notes in the Prompt 6 report)
- That padded multi-video batching is supported; `run_inference_batch` is sequential

## CLI

```bash
magic-vlm-infer --config configs/baseline_stub.yaml --example-id toy_train_001_q1
```

Refuses weight downloads unless `--allow-download` or a local checkpoint path is used.
