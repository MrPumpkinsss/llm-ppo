"""PPO-v2: PPO outputs device ordering, then DP decides layer allocation.

The RL agent learns to order devices optimally, and a DP solver handles
the precise layer allocation given that order. This combines the
exploration capability of RL with the optimality guarantees of DP
for the continuous partitioning subproblem.

Action space: Permutation of device indices (device ordering).
Observation space: Same as PPO-v1 (device properties, layer properties, bandwidth).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from config import TrainConfig
from environment import (
    DeviceCluster, LayerModel, create_random_config, compute_simple_tpot
)
from baselines import dp_for_device_order
from ppo_v1 import build_observation, get_obs_dim


class DeviceOrderNetwork(nn.Module):
    """Policy + value network for device ordering.

    Uses attention over devices to learn which ordering minimizes TPOT.
    The policy outputs a permutation (ordering) of devices.
    """

    def __init__(self, max_devices: int, obs_dim: int, hidden_dim: int = 256, num_layers_net: int = 4):
        super().__init__()
        self.max_devices = max_devices
        self.hidden_dim = hidden_dim
        self.num_layers_net = num_layers_net

        # Input projection for global observation
        self.fc_in = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Device feature encoder: 4 features per device (compute, avg_bw, max_bw, min_bw)
        self.device_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Deeper MLP for query/key/value computation
        self.query_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.key_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Permutation policy via pointer network style
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_value = nn.Linear(hidden_dim, 1)

        # Value head (deeper)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs: torch.Tensor, num_devices: int,
                device_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim) full observation
            num_devices: number of devices in this configuration
            device_features: (batch, num_devices, 4) per-device features

        Returns:
            order_logits: (batch, num_devices) logits for first selection
            value: (batch, 1)
        """
        B = obs.size(0)

        # Encode full observation
        x = self.fc_in(obs)  # (B, hidden)

        # Encode device features: (B, N, 4) -> (B, N, hidden)
        dev_encoded = self.device_encoder(device_features)  # (B, N, hidden)

        # Deeper query and key processing
        query = self.query_net(x)  # (B, hidden)
        keys = self.key_net(dev_encoded)  # (B, N, hidden)

        # Pointer network scores
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / (self.hidden_dim ** 0.5)

        value = self.value_head(x)

        return scores, value


def generate_device_order(
    logits: torch.Tensor, num_devices: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a device ordering using autoregressive sampling.

    The network outputs scores for up to max_devices. We use only the first
    num_devices entries and autoregressively mask already-selected devices.

    Args:
        logits: (max_devices,) scores for each device
        num_devices: number of actual devices

    Returns:
        order: (num_devices,) permutation of device indices
        log_prob: scalar
        entropy: scalar
    """
    # Slice to actual number of devices
    logits = logits[:num_devices]

    log_probs = []
    entropy = 0.0
    order = []
    available = list(range(num_devices))

    for step in range(num_devices):
        # Mask already selected devices
        mask = torch.full((num_devices,), -1e9, device=logits.device)
        for d in available:
            mask[d] = 0.0

        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        order.append(action.item())
        log_probs.append(dist.log_prob(action))
        entropy += dist.entropy()
        available.remove(action.item())

    return (
        torch.tensor(order, dtype=torch.long),
        torch.stack(log_probs).sum(),
        entropy
    )


def build_device_features(devices: DeviceCluster, num_devices: int) -> np.ndarray:
    """Build per-device feature vectors.

    Features per device:
    - compute power
    - average bandwidth to all other devices
    - max bandwidth to any device
    - min bandwidth to any device
    """
    features = np.zeros((num_devices, 4))
    for d in range(num_devices):
        features[d, 0] = devices.compute_power[d]
        bw_to_others = devices.bandwidth[d, np.arange(num_devices) != d]
        features[d, 1] = bw_to_others.mean() if len(bw_to_others) > 0 else 0
        features[d, 2] = bw_to_others.max() if len(bw_to_others) > 0 else 0
        features[d, 3] = bw_to_others.min() if len(bw_to_others) > 0 else 0

    return features.astype(np.float32)


def compute_reward_v2(
    device_order: list,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
    num_layers: int = 32,
) -> float:
    """Compute reward: negative TPOT of DP-optimal partition for given device order."""
    partition = dp_for_device_order(
        num_layers, device_order, devices, layers, tensor_size
    )
    tpot = compute_simple_tpot(partition, devices, layers, tensor_size)
    return -tpot


def log_prob_of_order(
    logits: torch.Tensor,
    order: torch.Tensor,
    num_devices: int,
) -> torch.Tensor:
    """Compute log probability of a given device order.

    Args:
        logits: (max_devices,) scores for each device
        order: (num_devices,) device ordering
    """
    logits = logits[:num_devices]
    order = order.to(logits.device)
    log_probs = []
    available = set(range(num_devices))

    for step in range(num_devices):
        mask = torch.full((num_devices,), -1e9, device=logits.device)
        for d in available:
            mask[d] = 0.0

        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs.append(dist.log_prob(order[step]))
        available.remove(order[step].item())

    return torch.stack(log_probs).sum()


def ppo_v2_inference(
    network: DeviceOrderNetwork,
    obs: torch.Tensor,
    device_features: torch.Tensor,
    num_devices: int,
    num_layers: int,
    devices,
    layers,
    tensor_size: float = 1.0,
) -> Tuple[list, float]:
    """PPO-v2 inference: try multiple orderings with DP auto-skip, return best partition.

    This function is the canonical PPO-v2 inference: PPO outputs device orderings,
    DP with auto-skip decides how many devices to actually use. Multiple candidate
    orderings are tried and the best TPOT is selected.

    Args:
        network: trained DeviceOrderNetwork
        obs: (1, obs_dim) observation tensor
        device_features: (1, num_devices, 4) per-device features
        num_devices: number of devices in this config
        num_layers: number of layers
        devices: DeviceCluster instance
        layers: LayerModel instance
        tensor_size: activation tensor size (GB)

    Returns:
        best_partition: optimal layer-to-device assignment
        best_tpot: TPOT of the best partition
    """
    from environment import compute_simple_tpot

    candidates = []

    with torch.no_grad():
        # 1. PPO greedy ordering
        order_logits, _ = network.forward(obs, num_devices, device_features)
        order, _, _ = generate_device_order(order_logits.squeeze(0), num_devices)
        order_list = order.tolist()
        part = dp_for_device_order(num_layers, order_list, devices, layers, tensor_size)
        candidates.append((compute_simple_tpot(part, devices, layers, tensor_size), part))

        # 2. Compute power descending
        sort_desc = sorted(range(num_devices), key=lambda d: devices.compute_power[d], reverse=True)
        part_desc = dp_for_device_order(num_layers, sort_desc, devices, layers, tensor_size)
        candidates.append((compute_simple_tpot(part_desc, devices, layers, tensor_size), part_desc))

        # 3. Compute power ascending
        sort_asc = sorted(range(num_devices), key=lambda d: devices.compute_power[d])
        part_asc = dp_for_device_order(num_layers, sort_asc, devices, layers, tensor_size)
        candidates.append((compute_simple_tpot(part_asc, devices, layers, tensor_size), part_asc))

        # 4. Top-3 PPO prefixes (PPO's top choices as first device)
        probs = torch.softmax(order_logits.squeeze(0)[:num_devices], dim=-1)
        top3 = torch.topk(probs, min(3, num_devices)).indices.tolist()
        for d1 in top3:
            remaining = [d for d in range(num_devices) if d != d1]
            for d2 in remaining[:3]:
                rest = [d for d in range(num_devices) if d not in [d1, d2]]
                order = [d1, d2] + rest
                part = dp_for_device_order(num_layers, order, devices, layers, tensor_size)
                candidates.append((compute_simple_tpot(part, devices, layers, tensor_size), part))

    best_tpot, best_partition = min(candidates, key=lambda x: x[0])
    return best_partition, best_tpot
