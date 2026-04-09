"""Configuration for LLM Layer Partitioning RL Environment."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Environment configuration."""
    num_layers: int = 32
    num_devices: int = 5
    device_compute_range: tuple = (0.1, 5.0) #(0.1, 1.0)
    layer_compute_range: tuple = (0.5, 2.0)
    bandwidth_range: tuple = (0.005, 3.0) # (0.05, 1.0)
    tensor_size: float = 1.0
    enforce_continuous: bool = True

    def __post_init__(self):
        assert self.num_layers >= 2, "Need at least 2 layers"
        assert self.num_devices >= 2, "Need at least 2 devices"


@dataclass
class TrainConfig:
    """Training configuration for all 5 versions."""

    # V1: DQN + min-max DP
    v1_learning_rate: float = 1e-3
    v1_hidden_dim: int = 256
    v1_batch_size: int = 128
    v1_gamma: float = 0.99
    v1_epsilon_start: float = 1.0
    v1_epsilon_end: float = 0.05
    v1_epsilon_decay: int = 600        # episodes over which to decay
    v1_target_update_freq: int = 200    # episodes between target net updates
    v1_replay_buffer_size: int = 10000
    v1_num_episodes: int = 10000
    v1_max_grad_norm: float = 1.0

    # V2: PPO binary device selection + min-max DP
    v2_learning_rate: float = 5e-4
    v2_hidden_dim: int = 256
    v2_batch_size: int = 256
    v2_epochs_per_update: int = 4
    v2_clip_eps: float = 0.2
    v2_gamma: float = 0.99
    v2_entropy_coef: float = 0.02
    v2_value_coef: float = 0.5
    v2_max_grad_norm: float = 0.5
    v2_num_episodes: int = 10000

    # V3: PPO-Clip one-shot ordering + min-max DP
    v3_learning_rate: float = 5e-4
    v3_hidden_dim: int = 256
    v3_batch_size: int = 256
    v3_epochs_per_update: int = 4
    v3_clip_eps: float = 0.2
    v3_gamma: float = 0.99
    v3_entropy_coef: float = 0.02
    v3_value_coef: float = 0.5
    v3_max_grad_norm: float = 0.5
    v3_num_episodes: int = 10000

    # V4: PPO-Clip autoregressive ordering + min-max DP
    v4_learning_rate: float = 5e-4
    v4_hidden_dim: int = 256
    v4_batch_size: int = 128
    v4_epochs_per_update: int = 4
    v4_clip_eps: float = 0.2
    v4_gamma: float = 0.99
    v4_gae_lambda: float = 0.95
    v4_entropy_coef: float = 0.02
    v4_value_coef: float = 0.5
    v4_max_grad_norm: float = 0.5
    v4_num_episodes: int = 10000

    # V5: Maskable PPO-Clip + min-max DP
    v5_learning_rate: float = 5e-4
    v5_hidden_dim: int = 256
    v5_batch_size: int = 128
    v5_epochs_per_update: int = 4
    v5_clip_eps: float = 0.2
    v5_gamma: float = 0.99
    v5_gae_lambda: float = 0.95
    v5_entropy_coef: float = 0.025
    v5_value_coef: float = 0.5
    v5_max_grad_norm: float = 0.5
    v5_num_episodes: int = 10000

    # V6: GNN-Based PPO + min-max DP
    v6_learning_rate: float = 5e-4
    v6_hidden_dim: int = 256
    v6_num_gnn_layers: int = 3
    v6_batch_size: int = 256
    v6_epochs_per_update: int = 4
    v6_clip_eps: float = 0.2
    v6_gamma: float = 0.99
    v6_entropy_coef: float = 0.02
    v6_value_coef: float = 0.5
    v6_max_grad_norm: float = 0.5
    v6_num_episodes: int = 10000

    # V7: Autoregressive GNN-PPO + Positional Encoding + min-max DP
    v7_learning_rate: float = 5e-4
    v7_hidden_dim: int = 256
    v7_num_gnn_layers: int = 3
    v7_batch_size: int = 128
    v7_epochs_per_update: int = 4
    v7_clip_eps: float = 0.2
    v7_gamma: float = 0.99
    v7_gae_lambda: float = 0.95
    v7_entropy_coef: float = 0.04
    v7_value_coef: float = 0.5
    v7_max_grad_norm: float = 0.5
    v7_num_episodes: int = 10000

    # Beam Search
    beam_width: int = 20

    # General
    seed: int = 42
    eval_interval: int = 500
    num_eval_configs: int = 50
    max_training_minutes: float = 15.0    # per version
    device: str = "cpu"
    train_v1: bool = False
    train_v2: bool = False
    train_v3: bool = True
    train_v4: bool = False
    train_v5: bool = False
    train_v6: bool = False
    train_v7: bool = False
    # Resume training
    resume_from_checkpoint: bool = False  # Continue from saved checkpoint
    checkpoint_dir: str = "results"       # Directory with saved models/metrics


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_test_configs: int = 50
    num_layers_to_test: list = field(default_factory=lambda: [16, 32, 48, 64])
    num_devices_to_test: list = field(default_factory=lambda: [3, 5, 7, 10])
    output_dir: str = "results"
    beam_width_eval: int = 5             # beam width for evaluation
    num_inference_candidates: int = 10   # best-of-N for RL models
