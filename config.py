"""Configuration for LLM Layer Partitioning RL Environment."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Environment configuration."""
    num_layers: int = 32            # Number of transformer layers
    num_devices: int = 5            # Number of heterogeneous devices
    device_compute_range: tuple = (0.1, 1.0)  # Device compute power range
    layer_compute_range: tuple = (0.5, 2.0)   # Layer compute cost range
    bandwidth_range: tuple = (0.5, 5.0)       # Inter-device bandwidth (GB/s)
    tensor_size: float = 1.0        # Activation tensor size (GB) per layer boundary
    # Continuous partition constraint: layers must be assigned in order
    enforce_continuous: bool = True

    def __post_init__(self):
        assert self.num_layers >= 2, "Need at least 2 layers"
        assert self.num_devices >= 2, "Need at least 2 devices"


@dataclass
class TrainConfig:
    """Training configuration."""
    # PPO-v1: direct layer assignment
    v1_learning_rate: float = 3e-4
    v1_hidden_dim: int = 256
    v1_num_layers: int = 4
    v1_batch_size: int = 256
    v1_epochs_per_update: int = 4
    v1_clip_eps: float = 0.2
    v1_gamma: float = 0.99
    v1_gae_lambda: float = 0.95
    v1_entropy_coef: float = 0.01
    v1_value_coef: float = 0.5
    v1_max_grad_norm: float = 0.5
    v1_num_episodes: int = 45000
    v1_steps_per_episode: int = 1  # single step per episode

    # PPO-v2: device ordering + DP
    v2_learning_rate: float = 5e-4
    v2_hidden_dim: int = 256
    v2_num_layers: int = 4
    v2_batch_size: float = 256
    v2_epochs_per_update: int = 4
    v2_clip_eps: float = 0.2
    v2_gamma: float = 0.99
    v2_gae_lambda: float = 0.95
    v2_entropy_coef: float = 0.01
    v2_value_coef: float = 0.5
    v2_max_grad_norm: float = 0.5
    v2_num_episodes: int = 45000
    v2_steps_per_episode: int = 1

    # General
    seed: int = 42
    eval_interval: int = 500
    num_eval_configs: int = 100     # Number of random configs per evaluation
    max_training_minutes: float = 120.0  # 2-hour hard limit
    device: str = "cuda"            # "cuda" or "cpu"

    @property
    def total_v1_updates(self):
        return self.v1_num_episodes // self.v1_batch_size + 1

    @property
    def total_v2_updates(self):
        return self.v2_num_episodes // self.v2_batch_size + 1


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_test_configs: int = 50
    num_layers_to_test: list = field(default_factory=lambda: [16, 32, 48, 64])
    num_devices_to_test: list = field(default_factory=lambda: [3, 5, 7, 10])
    output_dir: str = "results"
