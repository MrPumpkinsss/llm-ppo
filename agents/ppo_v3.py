"""V3: PPO-Clip + One-Shot Ordering + sum-based TPOT DP.

Single forward pass produces a complete device ordering via pointer-network
style attention. No autoregression -- the attention scores from one forward
pass are used to generate a permutation via sequential masking.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.shared import (
    MAX_DEVICES, MAX_LAYERS, get_obs_dim,
    build_observation, build_device_features,
)
from baselines import min_sum_tpot_dp
from environment import compute_simple_tpot


class PPOv3Network(nn.Module):
    """PPO-Clip network for one-shot device ordering via pointer attention.

    Observation: base obs (202-dim)
    Action: device permutation via attention scores
    """

    def __init__(self, obs_dim: int = None, hidden_dim: int = 256, max_devices: int = MAX_DEVICES):
        super().__init__()
        if obs_dim is None:
            obs_dim = get_obs_dim(max_devices, MAX_LAYERS)
        self.max_devices = max_devices
        self.hidden_dim = hidden_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

        # Device feature encoder
        self.device_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        # Pointer attention projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor, dev_features: torch.Tensor):
        """Forward pass.

        Args:
            obs: (B, obs_dim)
            dev_features: (B, max_devices, 4) per-device features

        Returns:
            attention_scores: (B, max_devices) raw scores for each device
            value: (B, 1) state value
        """
        obs_embed = self.obs_encoder(obs)           # (B, hidden)
        dev_embed = self.device_encoder(dev_features)  # (B, max_devices, hidden)

        query = self.query_proj(obs_embed)           # (B, hidden)
        keys = self.key_proj(dev_embed)              # (B, max_devices, hidden)

        # Dot-product attention: (B, max_devices)
        attention_scores = torch.bmm(
            keys, query.unsqueeze(2)
        ).squeeze(2) / np.sqrt(self.hidden_dim)

        value = self.value_head(obs_embed)
        return attention_scores, value


def ppo_v3_generate_ordering(
    network: PPOv3Network,
    obs: torch.Tensor,
    dev_features: torch.Tensor,
    num_devices: int,
    deterministic: bool = False,
    temperature: float = 1.0,
):
    """Generate device ordering from attention scores via sequential masking.

    Returns:
        ordering: list of device indices in pipeline order
        log_prob: total log probability of the ordering
        attention_scores: raw scores
        value: state value
    """
    attention_scores, value = network(obs, dev_features)

    # Sequential masking to produce a permutation
    B = obs.shape[0]
    ordering_batch = []
    log_prob_batch = []

    for b in range(B):
        scores = attention_scores[b].clone()
        # Mask out devices beyond num_devices
        mask = torch.zeros(network.max_devices, device=obs.device, dtype=torch.bool)
        mask[:num_devices] = True

        ordering = []
        total_log_prob = 0.0

        for step in range(num_devices):
            # Apply mask
            masked_scores = scores.clone()
            masked_scores[~mask] = float('-inf')

            # Temperature-scaled softmax
            probs = F.softmax(masked_scores / temperature, dim=0)

            if deterministic:
                action = probs.argmax().item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

            log_p = torch.log(probs[action].clamp(min=1e-8)).item()
            total_log_prob += log_p

            ordering.append(action)
            mask[action] = False

        ordering_batch.append(ordering)
        log_prob_batch.append(total_log_prob)

    return ordering_batch, torch.FloatTensor(log_prob_batch).to(obs.device), attention_scores, value


def ppo_v3_inference(
    network: PPOv3Network,
    devices,
    layers,
    tensor_size: float,
    num_layers: int,
    num_devices: int,
    torch_device: torch.device,
    num_candidates: int = 10,
):
    """Pure greedy inference — single deterministic pass."""
    network.eval()

    base_obs = build_observation(devices, layers, tensor_size, num_layers, num_devices)
    dev_feats = build_device_features(devices, num_devices)
    obs_tensor = torch.FloatTensor(base_obs).unsqueeze(0).to(torch_device)
    dev_tensor = torch.FloatTensor(dev_feats).unsqueeze(0).to(torch_device)

    with torch.no_grad():
        orderings, _, _, _ = ppo_v3_generate_ordering(
            network, obs_tensor, dev_tensor, num_devices, deterministic=True
        )

    ordering = orderings[0]
    partition = min_sum_tpot_dp(num_layers, ordering, devices, layers, tensor_size)
    tpot = compute_simple_tpot(partition, devices, layers, tensor_size)

    return partition, tpot
