"""V4: PPO-Clip + Autoregressive Ordering + min-max DP.

GRU-based autoregressive device selection. At each step, the network
picks the next device from remaining devices. The ordering goes to
min_max_bottleneck_dp for layer allocation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.shared import (
    MAX_DEVICES, MAX_LAYERS, get_seq_obs_dim,
    build_observation, build_sequential_observation,
)
from baselines import min_max_bottleneck_dp
from environment import compute_simple_tpot


class PPOv4Network(nn.Module):
    """PPO-Clip network with GRU for autoregressive device ordering.

    Observation: sequential obs (220-dim) per step
    Action: Discrete(MAX_DEVICES) — pick next device from remaining
    """

    def __init__(self, obs_dim: int = None, hidden_dim: int = 256, max_devices: int = MAX_DEVICES):
        super().__init__()
        if obs_dim is None:
            obs_dim = get_seq_obs_dim(max_devices, MAX_LAYERS)
        self.hidden_dim = hidden_dim
        self.max_devices = max_devices

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_devices),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """Forward pass for one step.

        Args:
            obs: (B, obs_dim)
            hidden: (B, hidden_dim)

        Returns:
            logits: (B, max_devices)
            value: (B, 1)
            new_hidden: (B, hidden_dim)
        """
        encoded = self.obs_encoder(obs)
        h = self.gru(encoded, hidden)
        logits = self.action_head(h)
        value = self.value_head(h)
        return logits, value, h

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


def ppo_v4_generate_episode(
    network: PPOv4Network,
    devices,
    layers,
    tensor_size: float,
    num_layers: int,
    num_devices: int,
    device: torch.device,
    deterministic: bool = False,
    temperature: float = 1.0,
):
    """Generate one autoregressive episode.

    Returns:
        step_data: list of (obs, action, log_prob, value, mask) per step
        ordering: final device ordering
        partition: final partition
        tpot: TPOT
    """
    network.eval()
    base_obs = build_observation(devices, layers, tensor_size, num_layers, num_devices)
    selected = []
    selected_mask = np.zeros(MAX_DEVICES, dtype=np.float32)
    hidden = network.init_hidden(1, device)
    step_data = []

    for step in range(num_devices):
        seq_obs = build_sequential_observation(base_obs, selected_mask, step, num_devices)
        obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(device)

        # Store hidden state BEFORE this step's forward pass
        h_before = hidden.clone()

        with torch.no_grad():
            logits, value, hidden = network(obs_tensor, hidden)

        # Build mask: only valid (unselected, within num_devices) devices
        mask = np.zeros(MAX_DEVICES, dtype=bool)
        for d in range(num_devices):
            if d not in selected:
                mask[d] = True

        if not mask.any():
            break

        masked_logits = logits.clone()
        masked_logits[0, ~torch.BoolTensor(mask).to(device)] = float('-inf')

        if temperature != 1.0:
            masked_logits = masked_logits / temperature

        probs = F.softmax(masked_logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

        log_prob = torch.log(probs[0, action].clamp(min=1e-8)).item()

        step_data.append((
            seq_obs, action, log_prob, value.item(), mask.copy(),
            h_before.squeeze(0).cpu().numpy()
        ))

        selected.append(action)
        selected_mask[action] = 1.0

    # Compute final partition and reward
    ordering = selected if selected else [0]
    partition = min_max_bottleneck_dp(num_layers, ordering, devices, layers, tensor_size)
    tpot = compute_simple_tpot(partition, devices, layers, tensor_size)

    return step_data, ordering, partition, tpot


def ppo_v4_inference(
    network: PPOv4Network,
    devices,
    layers,
    tensor_size: float,
    num_layers: int,
    num_devices: int,
    torch_device: torch.device,
    num_candidates: int = 10,
):
    """Pure greedy inference — single deterministic pass."""
    _, ordering, partition, tpot = ppo_v4_generate_episode(
        network, devices, layers, tensor_size, num_layers, num_devices,
        torch_device, deterministic=True
    )
    return partition, tpot
