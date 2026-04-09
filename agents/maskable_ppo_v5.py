"""V5: Maskable PPO-Clip + Sequential Device Selection + min-max DP.

PPO selects devices one-at-a-time from unused pool, or chooses STOP.
Not all devices need to be used. Invalid-action masking is handled properly:
masks are stored with transitions and replayed exactly during PPO updates.
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

STOP_ACTION = MAX_DEVICES  # action index for STOP


class MaskablePPOv5Network(nn.Module):
    """Maskable PPO-Clip network with GRU for sequential device selection.

    Observation: sequential obs (220-dim) per step
    Action: Discrete(MAX_DEVICES + 1) — pick device 0-15 or STOP (action 16)
    """

    def __init__(self, obs_dim: int = None, hidden_dim: int = 256, max_devices: int = MAX_DEVICES):
        super().__init__()
        if obs_dim is None:
            obs_dim = get_seq_obs_dim(max_devices, MAX_LAYERS)
        self.hidden_dim = hidden_dim
        self.max_devices = max_devices
        self.num_actions = max_devices + 1  # devices + STOP

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
            nn.Linear(128, self.num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor, action_mask: torch.Tensor):
        """Forward pass with action masking.

        Args:
            obs: (B, obs_dim)
            hidden: (B, hidden_dim)
            action_mask: (B, num_actions) bool, True=valid

        Returns:
            masked_logits: (B, num_actions)
            value: (B, 1)
            new_hidden: (B, hidden_dim)
        """
        encoded = self.obs_encoder(obs)
        h = self.gru(encoded, hidden)
        logits = self.action_head(h)
        # Apply mask
        masked_logits = logits.masked_fill(~action_mask, float('-inf'))
        value = self.value_head(h)
        return masked_logits, value, h

    def forward_no_mask(self, obs: torch.Tensor, hidden: torch.Tensor):
        """Forward pass without masking (for value computation during update)."""
        encoded = self.obs_encoder(obs)
        h = self.gru(encoded, hidden)
        logits = self.action_head(h)
        value = self.value_head(h)
        return logits, value, h

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


def build_action_mask(selected_mask: np.ndarray, num_devices: int) -> np.ndarray:
    """Build action mask for V5.

    Args:
        selected_mask: (MAX_DEVICES,) 1.0 = already selected
        num_devices: total number of devices

    Returns:
        mask: (num_actions,) bool, True=valid action
    """
    mask = np.zeros(STOP_ACTION + 1, dtype=bool)
    # Devices that are within num_devices and not yet selected
    for d in range(num_devices):
        if selected_mask[d] < 0.5:  # not selected
            mask[d] = True
    # Devices beyond num_devices are always invalid
    for d in range(num_devices, MAX_DEVICES):
        mask[d] = False
    # STOP is always valid (can stop at any time)
    mask[STOP_ACTION] = True
    return mask


def maskable_v5_generate_episode(
    network: MaskablePPOv5Network,
    devices,
    layers,
    tensor_size: float,
    num_layers: int,
    num_devices: int,
    device: torch.device,
    deterministic: bool = False,
    temperature: float = 1.0,
):
    """Generate one episode with Maskable PPO.

    Returns:
        step_data: list of (obs, action, log_prob, value, action_mask) per step
        ordering: device ordering (without STOP)
        partition: final partition
        tpot: TPOT
    """
    network.eval()
    base_obs = build_observation(devices, layers, tensor_size, num_layers, num_devices)
    selected = []
    selected_mask = np.zeros(MAX_DEVICES, dtype=np.float32)
    hidden = network.init_hidden(1, device)
    step_data = []

    for step in range(MAX_DEVICES + 1):  # max steps = all devices + 1
        action_mask = build_action_mask(selected_mask, num_devices)

        # If no valid device actions and only STOP, force STOP
        valid_devices = [d for d in range(num_devices) if d not in selected]
        if not valid_devices:
            # All devices selected, must STOP
            break

        seq_obs = build_sequential_observation(base_obs, selected_mask, step, num_devices)
        obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(device)
        mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(device)

        # Store hidden state BEFORE this step's forward pass
        h_before = hidden.clone()

        with torch.no_grad():
            masked_logits, value, hidden = network(obs_tensor, hidden, mask_tensor)

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
            seq_obs, action, log_prob, value.item(), action_mask.copy(),
            h_before.squeeze(0).cpu().numpy()
        ))

        if action == STOP_ACTION:
            break

        # Select device
        selected.append(action)
        selected_mask[action] = 1.0

    # Compute final partition
    ordering = selected if selected else [0]
    partition = min_max_bottleneck_dp(num_layers, ordering, devices, layers, tensor_size)
    tpot = compute_simple_tpot(partition, devices, layers, tensor_size)

    return step_data, ordering, partition, tpot


def maskable_v5_inference(
    network: MaskablePPOv5Network,
    devices,
    layers,
    tensor_size: float,
    num_layers: int,
    num_devices: int,
    torch_device: torch.device,
    num_candidates: int = 10,
):
    """Pure greedy inference — single deterministic pass."""
    _, ordering, partition, tpot = maskable_v5_generate_episode(
        network, devices, layers, tensor_size, num_layers, num_devices,
        torch_device, deterministic=True
    )
    return partition, tpot
