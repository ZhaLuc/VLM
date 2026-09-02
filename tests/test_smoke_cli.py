"""End-to-end interface smoke without downloading weights."""

from pathlib import Path

from magic_vlm.cli import smoke_main


def test_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    # Redirect outputs away from the repo default runs/ during tests.
    monkeypatch.chdir(Path.cwd())
    code = smoke_main(["--config", "configs/baseline_stub.yaml"])
    assert code == 0
    runs = Path("runs")
    assert runs.exists()
    manifests = list(runs.glob("*/run_manifest.json"))
    preds = list(runs.glob("*/predictions.jsonl"))
    assert manifests
    assert preds
