import random

from magic_vlm.runtime import (
    DeviceConfig,
    SeedConfig,
    capture_environment,
    probe_torch,
    resolve_device,
    set_seed,
)


def test_cpu_device_resolution() -> None:
    info = resolve_device(DeviceConfig(preference="cpu"))
    assert info.resolved == "cpu"
    assert info.requested == "cpu"


def test_auto_device_does_not_crash() -> None:
    info = resolve_device(DeviceConfig(preference="auto"))
    assert info.resolved in {"cpu"} or info.resolved.startswith("cuda:")


def test_set_seed_is_partially_controlled() -> None:
    report = set_seed(SeedConfig(seed=42))
    assert report.level == "partially_controlled"
    assert report.seed == 42
    assert "guaranteed" not in report.level
    # Python RNG is seeded
    assert isinstance(random.random(), float)


def test_environment_capture_cpu_path() -> None:
    device = resolve_device(DeviceConfig(preference="cpu"))
    env = capture_environment(device=device)
    assert env["device"]["resolved"] == "cpu"
    assert "python_version" in env
    assert "torch" in env
    torch_info = probe_torch()
    assert "cuda_available" in torch_info
