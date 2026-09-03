"""Magic/mentalism VLM research package.

Minimal architecture for dataset schemas, deterministic video preprocessing,
model loading, inference (with raw-output preservation), evaluation, preference
records, standalone rewards, experiment configuration, and reproducibility
metadata.

Training algorithms: DPO (``magic_vlm.dpo``) and GRPO on modular objective
rewards (``magic_vlm.grpo``, TRL + optional PEFT). PPO on the VLM is not
implemented. A Bradley-Terry preference reward model is separate.
"""

from magic_vlm.schemas import ExampleRecord, Split, TaskType, VideoRef
from magic_vlm.experiment import ExperimentConfig, initialize_experiment, load_experiment_config

__all__ = [
    "ExampleRecord",
    "Split",
    "TaskType",
    "VideoRef",
    "ExperimentConfig",
    "load_experiment_config",
    "initialize_experiment",
]

__version__ = "0.1.19"
