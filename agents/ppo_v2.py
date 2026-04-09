"""V2: PPO + Binary Device Selection + min-max DP.

Single-step PPO. One forward pass outputs per-device selection probability.
Selected devices sorted by compute power -> min_max_bottleneck_dp.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.shared import (
    MAX_DEVICES, MAX_LAYERS, get_obs_dim,
    build_observation,
)
from baselines import min_max_bottleneck_dp
from environment import compute_simple_tpot


class PPOv2Network(nn.Module):
    """PPO network for binary device selection.

    Observation: base obs (202-dim)
    Action: Multi-binary over MAX_DEVICES devices (sigmoid logits -> Bernoulli)
    """

    def __init__(self, obs_dim: int = None, hidden_dim: int = 256, max_devices: int = MAX_DEVICES):
        super().__init__()
        if obs_dim is None:
            obs_dim = get_obs_dim(max_devices, MAX_LAYERS)
        self.max_devices = max_devices

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )
        self.device_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_devices),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor):
        """Forward pass.

        Args:
            obs: (B, obs_dim)

        Returns:
            device_logits: (B, max_devices) raw logits for each device
            value: (B, 1) state value
        """
        x = self.encoder(obs)
        device_logits = self.device_head(x)
        value = self.value_head(x)
        return device_logits, value


def ppo_v2_sample_action(
    network: PPOv2Network,
    obs: torch.Tensor,
    num_devices: int,
    deterministic: bool = False,
):
    """Sample binary device selection.

    Returns:
        selection: (B, max_devices) binary tensor
        log_probs: (B,) log probability of the selection
        device_logits: raw logits
        value: (B, 1)
    """
    device_logits, value = network(obs)
    probs = torch.sigmoid(device_logits.clamp(-20, 20))
    probs = torch.nan_to_num(probs, nan=0.5).clamp(0, 1)

    # Mask out devices beyond num_devices
    mask = torch.zeros_like(probs)
    mask[:, :num_devices] = 1.0
    probs = probs * mask

    if deterministic:
        selection = (probs > 0.5).float()
    else:
        selection = torch.bernoulli(probs)

    # Ensure at least 1 device selected
    none_selected = (selection[:, :num_devices].sum(dim=1) == 0)
    if none_selected.any():
        # Force the device with highest probability
        best_dev = probs[:, :num_devices].argmax(dim=1)
        for i in range(selection.shape[0]):
            if none_selected[i]:
                selection[i, best_dev[i]] = 1.0

    # Compute log probability: sum of Bernoulli log probs
    selected_probs = probs * selection + (1 - probs) * (1 - selection)
    log_probs = torch.log(selected_probs[:, :num_devices].clamp(min=1e-8)).sum(dim=1)

    return selection, log_probs, device_logits, value


def selection_to_partition(
    selection: np.ndarray,
    num_layers: int,
    num_devices: int,
    devices,
    layers,
    tensor_size: float,
) -> list:
    """Convert binary selection to partition via min-max DP.

    Args:
        selection: (max_devices,) binary array
        num_layers, num_devices, devices, layers, tensor_size: env params

    Returns:
        partition: list of device indices per layer
    """
    selected_devices = [d for d in range(num_devices) if selection[d] > 0.5]
    if not selected_devices:
        selected_devices = [0]

    # Sort selected devices by compute power (descending) for better DP
    selected_devices.sort(key=lambda d: -devices.compute_power[d])

    partition = min_max_bottleneck_dp(
        num_layers, selected_devices, devices, layers, tensor_size
    )
    return partition


def ppo_v2_inference(
    network: PPOv2Network,
    devices,
    layers,
    tensor_size: float,
    num_layers: int,
    num_devices: int,
    torch_device: torch.device,
    num_candidates: int = 10,
):
    """Pure greedy inference — single deterministic pass, threshold=0.5."""
    network.eval()

    base_obs = build_observation(devices, layers, tensor_size, num_layers, num_devices)
    obs_tensor = torch.FloatTensor(base_obs).unsqueeze(0).to(torch_device)

    with torch.no_grad():
        device_logits, _ = network(obs_tensor)
        probs = torch.sigmoid(device_logits).cpu().numpy().flatten()

    sel = np.zeros(MAX_DEVICES, dtype=np.float32)
    sel[:num_devices] = (probs[:num_devices] > 0.5).astype(np.float32)
    if sel[:num_devices].sum() == 0:
        sel[np.argmax(probs[:num_devices])] = 1.0
    partition = selection_to_partition(sel, num_layers, num_devices, devices, layers, tensor_size)
    tpot = compute_simple_tpot(partition, devices, layers, tensor_size)

    return partition, tpot
