"""Device selection, seeding, and environment capture.

Determinism policy (research integrity):
- Default level is ``partially_controlled`` after seeds are applied.
- ``guaranteed`` bitwise determinism is **not** claimed and is never reported
  by this module without an explicit verified harness (not implemented here).
- CUDA/cuDNN nondeterminism, Hugging Face generate kernels, and data-loader
  ordering can still change scientific results even when seeds match.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from magic_vlm.utils import git_commit_sha, utc_now_iso

DeterminismLevel = Literal["unavailable", "partially_controlled", "guaranteed"]


@dataclass(frozen=True)
class DeviceConfig:
    """Device preference without hard-coded machine-specific GPU IDs.

    ``preference``:
      - ``auto``: use CUDA if a CUDA-capable PyTorch build sees GPUs, else CPU
      - ``cpu``: force CPU
      - ``cuda``: require CUDA; optional ``cuda_index`` selects ``cuda:N``
    """

    preference: Literal["auto", "cpu", "cuda"] = "auto"
    cuda_index: int | None = None

    def __post_init__(self) -> None:
        if self.cuda_index is not None and self.cuda_index < 0:
            raise ValueError("cuda_index must be >= 0 when provided")


@dataclass(frozen=True)
class SeedConfig:
    """Random-seed and optional library determinism knobs."""

    seed: int = 0
    deterministic_algorithms: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    resolved: str
    torch_available: bool
    cuda_available: bool
    cuda_device_count: int
    device_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeterminismReport:
    """Honest status of reproducibility controls actually applied."""

    level: DeterminismLevel
    seed: int
    settings_applied: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


def probe_torch() -> dict[str, Any]:
    """Return torch/CUDA facts without selecting a device."""
    try:
        import torch  # type: ignore
    except ImportError:
        return {
            "torch_available": False,
            "torch_version": None,
            "cuda_compiled": False,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_version": None,
            "cudnn_version": None,
            "device_names": [],
        }

    cuda_available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda_available else 0
    names = []
    if cuda_available:
        for index in range(count):
            names.append(torch.cuda.get_device_name(index))
    return {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_compiled": bool(getattr(torch.version, "cuda", None)),
        "cuda_available": cuda_available,
        "cuda_device_count": count,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if cuda_available and torch.backends.cudnn.is_available()
            else None
        ),
        "device_names": names,
    }


def resolve_device(config: DeviceConfig | None = None) -> DeviceInfo:
    """Resolve a portable device string (``cpu`` or ``cuda[:index]``)."""
    cfg = config or DeviceConfig()
    torch_info = probe_torch()
    torch_available = bool(torch_info["torch_available"])
    cuda_available = bool(torch_info["cuda_available"])
    count = int(torch_info["cuda_device_count"])

    if cfg.preference == "cpu":
        return DeviceInfo(
            requested="cpu",
            resolved="cpu",
            torch_available=torch_available,
            cuda_available=cuda_available,
            cuda_device_count=count,
            device_name="cpu",
        )

    if cfg.preference == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "Device preference is 'cuda' but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or set device.preference: cpu."
            )
        index = 0 if cfg.cuda_index is None else cfg.cuda_index
        if index >= count:
            raise RuntimeError(
                f"Requested cuda_index={index} but only {count} CUDA device(s) are visible"
            )
        name = torch_info["device_names"][index] if torch_info["device_names"] else None
        return DeviceInfo(
            requested=f"cuda:{index}" if cfg.cuda_index is not None else "cuda",
            resolved=f"cuda:{index}",
            torch_available=True,
            cuda_available=True,
            cuda_device_count=count,
            device_name=name,
        )

    # auto
    if cuda_available and count > 0:
        index = 0 if cfg.cuda_index is None else cfg.cuda_index
        if index >= count:
            raise RuntimeError(
                f"Requested cuda_index={index} but only {count} CUDA device(s) are visible"
            )
        name = torch_info["device_names"][index] if torch_info["device_names"] else None
        return DeviceInfo(
            requested="auto",
            resolved=f"cuda:{index}",
            torch_available=True,
            cuda_available=True,
            cuda_device_count=count,
            device_name=name,
        )
    return DeviceInfo(
        requested="auto",
        resolved="cpu",
        torch_available=torch_available,
        cuda_available=cuda_available,
        cuda_device_count=count,
        device_name="cpu",
    )


def set_seed(config: SeedConfig | int) -> DeterminismReport:
    """Apply process-level seeds. Does **not** claim bitwise run reproducibility."""
    cfg = SeedConfig(seed=config) if isinstance(config, int) else config
    notes: list[str] = [
        "Seeds applied for Python random and (if present) NumPy/Torch RNGs.",
        "Bitwise identical VLM outputs across hardware/library versions are not guaranteed.",
    ]
    settings: dict[str, Any] = {
        "python_random_seeded": True,
        "numpy_seeded": False,
        "torch_seeded": False,
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "python_hash_seed_env": os.environ.get("PYTHONHASHSEED"),
    }

    random.seed(cfg.seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(cfg.seed)
        settings["numpy_seeded"] = True
    except ImportError:
        notes.append("NumPy not installed; skipped np.random.seed.")

    try:
        import torch  # type: ignore

        torch.manual_seed(cfg.seed)
        settings["torch_seeded"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
            settings["torch_cuda_manual_seed_all"] = True
        if cfg.deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
            settings["torch_deterministic_algorithms"] = True
            notes.append(
                "torch.use_deterministic_algorithms(True) requested; some ops may error."
            )
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = bool(cfg.cudnn_deterministic)
            torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
            settings["cudnn_deterministic"] = bool(cfg.cudnn_deterministic)
            settings["cudnn_benchmark"] = bool(cfg.cudnn_benchmark)
    except ImportError:
        notes.append("PyTorch not installed; skipped torch seeding.")

    if os.environ.get("PYTHONHASHSEED") is None:
        notes.append(
            "PYTHONHASHSEED is unset; set it in the process environment before "
            "interpreter start for hash-order stability."
        )

    level: DeterminismLevel = "partially_controlled"
    return DeterminismReport(
        level=level,
        seed=cfg.seed,
        settings_applied=settings,
        notes=tuple(notes),
    )


def capture_environment(*, device: DeviceInfo | None = None) -> dict[str, Any]:
    """Serializable runtime snapshot for experiment metadata."""
    resolved = device or resolve_device(DeviceConfig(preference="auto"))
    torch_info = probe_torch()
    return {
        "captured_at": utc_now_iso(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "package_version": _package_version(),
        "git_commit": git_commit_sha(),
        "git_dirty": _git_dirty(),
        "device": resolved.to_dict(),
        "torch": torch_info,
        "env_subset": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }


def _package_version() -> str:
    try:
        from magic_vlm import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_dirty() -> bool | None:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())
