"""Magic/mentalism VLM research package.

Minimal architecture for dataset schemas, deterministic video preprocessing,
model loading, inference (with raw-output preservation), evaluation, preference
records, standalone rewards, and experiment configuration.

Training algorithms (DPO/GRPO/PPO) and reward-model fitting are intentionally
not implemented in this stage.
"""

from magic_vlm.schemas import ExampleRecord, Split, VideoRef
from magic_vlm.experiment import ExperimentConfig, load_experiment_config

__all__ = [
    "ExampleRecord",
    "Split",
    "VideoRef",
    "ExperimentConfig",
    "load_experiment_config",
]

__version__ = "0.1.0"
