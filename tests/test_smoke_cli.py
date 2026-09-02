from pathlib import Path

from magic_vlm.cli import init_main, smoke_main
from magic_vlm.inference import GenerationConfig


def test_generation_config_serialization() -> None:
    gen = GenerationConfig(max_new_tokens=16, temperature=0.7, top_p=0.9, top_k=50, do_sample=True)
    payload = gen.to_dict()
    assert payload["sampling_mode"] == "sample"
    assert payload["top_p"] == 0.9
    roundtrip = GenerationConfig.from_dict(payload)
    assert roundtrip.do_sample is True
    assert roundtrip.top_k == 50


def test_init_main_no_model_load(tmp_path: Path, monkeypatch) -> None:
    # Point config output into temp by rewriting via monkeypatch of cwd files is heavy;
    # invoke initialize path through CLI after copying config.
    cfg_text = Path("configs/baseline_stub.yaml").read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("output_dir: runs", f"output_dir: {tmp_path.as_posix()}/runs")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    code = init_main(["--config", str(cfg_path), "--run-id", "init-smoke"])
    assert code == 0
    run_dir = tmp_path / "runs" / "init-smoke"
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "environment.json").exists()
    assert not (run_dir / "predictions.jsonl").exists()


def test_cli_smoke(tmp_path: Path) -> None:
    cfg_text = Path("configs/baseline_stub.yaml").read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("output_dir: runs", f"output_dir: {tmp_path.as_posix()}/runs")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    code = smoke_main(["--config", str(cfg_path), "--run-id", "smoke-run"])
    assert code == 0
    run_dir = tmp_path / "runs" / "smoke-run"
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "predictions.jsonl").exists()
