"""Base trainer with shared training infrastructure."""
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

from environment import DeviceCluster, LayerModel, create_random_config, compute_simple_tpot
from baselines import dp_partition, min_max_bottleneck_dp, beam_search_dp


@dataclass
class TrainingMetrics:
    """Tracks training progress."""
    episode_rewards: list = field(default_factory=list)
    episode_tpot: list = field(default_factory=list)
    policy_losses: list = field(default_factory=list)
    value_losses: list = field(default_factory=list)
    entropies: list = field(default_factory=list)
    eval_rewards: list = field(default_factory=list)
    eval_tpot: list = field(default_factory=list)
    dp_tpot: list = field(default_factory=list)
    beam_tpot: list = field(default_factory=list)
    greedy_tpot: list = field(default_factory=list)
    wall_time: list = field(default_factory=list)
    episodes_log: list = field(default_factory=list)


class BaseTrainer:
    """Shared training infrastructure for all 5 versions."""

    def __init__(self, config, version_name: str, max_minutes: float = 5.0):
        self.config = config
        self.version_name = version_name
        self.max_minutes = max_minutes
        self.metrics = TrainingMetrics()
        self.start_time = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.best_eval_reward = float('-inf')
        self.best_network_state = None

    def _create_random_env(self, seed=None, max_nd=None, max_nl=None):
        """Generate random environment config with random (nl, nd)."""
        if seed is None:
            seed = np.random.randint(0, 100000)
        rng = np.random.RandomState(seed)

        max_nl = max_nl or 64
        max_nd = max_nd or 10
        nl = rng.randint(8, max_nl + 1)
        nd = rng.randint(2, max_nd + 1)
        # Round nl to multiple of 4 for cleaner partitions
        nl = max(8, (nl // 4) * 4)

        return create_random_config(nl, nd, seed=seed)

    def _compute_baselines(self, devices, layers, ts, nl, nd):
        """Compute DP TPOT for reward shaping. Beam search is too slow for training."""
        try:
            dp_part = dp_partition(nl, nd, devices, layers, ts)
            dp_tpot = compute_simple_tpot(dp_part, devices, layers, ts)
        except Exception:
            dp_tpot = float('inf')

        return dp_tpot, None  # No beam search during training

    def _evaluate(self, network, step, num_configs=30):
        """Evaluate the network on random configs."""
        was_training = network.training
        network.eval()

        rewards = []
        tpots = []
        dp_tpots = []
        beam_tpots = []

        for i in range(num_configs):
            devices, layers, ts = self._create_random_env(seed=99900 + i)
            nl, nd = layers.num_layers, devices.num_devices

            dp_tpot, beam_tpot = self._compute_baselines(devices, layers, ts, nl, nd)

            partition, tpot = self._inference_one(network, devices, layers, ts, nl, nd)

            from agents.shared import compute_reward
            reward = compute_reward(partition, devices, layers, ts)

            rewards.append(reward)
            tpots.append(tpot)
            dp_tpots.append(dp_tpot)
            if beam_tpot is not None:
                beam_tpots.append(beam_tpot)

        avg_reward = np.mean(rewards)
        self.metrics.eval_rewards.append(avg_reward)
        self.metrics.eval_tpot.append(np.mean(tpots))
        self.metrics.dp_tpot.append(np.mean(dp_tpots))
        self.metrics.beam_tpot.append(np.mean(beam_tpots) if beam_tpots else 0.0)
        self.metrics.episodes_log.append(step)

        # Save best checkpoint
        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            self.best_network_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}

        if was_training:
            network.train()

        return avg_reward

    def _inference_one(self, network, devices, layers, ts, nl, nd):
        """Run inference for one config. Override in subclasses."""
        raise NotImplementedError

    def _check_time_budget(self):
        """Return True if time budget exceeded."""
        if self.start_time is None:
            return False
        elapsed = (time.time() - self.start_time) / 60.0
        return elapsed >= self.max_minutes

    def _log_progress(self, episode, total, extra_info=""):
        elapsed = time.time() - self.start_time if self.start_time else 0
        metrics = self.metrics
        if metrics.episode_rewards:
            avg_r = np.mean(metrics.episode_rewards[-50:])
            avg_t = np.mean(metrics.episode_tpot[-50:])
            print(f"  [{self.version_name}] Ep {episode}/{total} | "
                  f"AvgR: {avg_r:.2f} | AvgTPOT: {avg_t:.3f} | "
                  f"Time: {elapsed:.0f}s {extra_info}")

    def train(self):
        """Main training loop. Override in subclasses."""
        raise NotImplementedError

    def _restore_best(self, network):
        """Restore best network from checkpoint."""
        if self.best_network_state is not None:
            network.load_state_dict(self.best_network_state)
            network.to(self.device)


def ppo_clip_update(
    network, optimizer,
    obs_batch, action_batch, old_log_prob_batch,
    return_batch, advantage_batch,
    clip_eps=0.2, value_coef=0.5, entropy_coef=0.02, max_grad_norm=0.5,
    mask_batch=None, num_actions=None,
):
    """Standard PPO-Clip update (shared by V2-V5).

    Args:
        network: the policy network
        optimizer: optimizer
        obs_batch: (B, obs_dim)
        action_batch: (B,)
        old_log_prob_batch: (B,)
        return_batch: (B,)
        advantage_batch: (B,)
        clip_eps: clipping parameter
        value_coef: value loss coefficient
        entropy_coef: entropy bonus coefficient
        max_grad_norm: max gradient norm
        mask_batch: (B, num_actions) optional action masks for maskable PPO
        num_actions: number of actions (for entropy normalization)

    Returns:
        policy_loss, value_loss, entropy
    """
    obs_tensor = torch.FloatTensor(obs_batch)
    action_tensor = torch.LongTensor(action_batch)
    old_log_probs = torch.FloatTensor(old_log_prob_batch)
    returns = torch.FloatTensor(return_batch)
    advantages = torch.FloatTensor(advantage_batch)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Forward pass - depends on network type
    # This will be handled by the caller passing in pre-computed values
    # or we use a generic interface

    # For V2: network returns (device_logits, value)
    # For V3: network returns (attention_scores, value) with dev_features
    # For V4/V5: network returns (logits, value, hidden) with mask

    # Generic: call network-specific forward and compute losses
    raise NotImplementedError("Use version-specific update functions")
