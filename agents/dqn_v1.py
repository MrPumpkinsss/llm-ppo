"""V1: DQN + min-max bottleneck DP.

DQN selects devices autoregressively (one at a time or STOP),
forming an ordered list. The ordered list goes to min_max_bottleneck_dp
for optimal continuous layer allocation.
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

from agents.shared import (
    MAX_DEVICES, MAX_LAYERS, get_seq_obs_dim,
    build_observation, build_sequential_observation,
)
from baselines import min_max_bottleneck_dp
from environment import compute_simple_tpot


class ReplayBuffer:
    """Simple circular replay buffer for DQN with hidden states."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, hidden_state):
        self.buffer.append((state, action, reward, next_state, done, hidden_state))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, hiddens = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(hiddens, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNv1Network(nn.Module):
    """DQN network with GRU for autoregressive device selection.

    Observation: sequential obs (base + selected_mask + step_norm) = 220-dim
    Action: Discrete(17) — select device 0-15 or STOP (action 16)
    """

    def __init__(self, obs_dim: int = None, hidden_dim: int = 256, max_devices: int = MAX_DEVICES):
        super().__init__()
        if obs_dim is None:
            obs_dim = get_seq_obs_dim(max_devices, MAX_LAYERS)
        self.hidden_dim = hidden_dim
        self.max_devices = max_devices
        self.num_actions = max_devices + 1  # 16 devices + STOP

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """Forward pass.

        Args:
            obs: (B, obs_dim)
            hidden: (B, hidden_dim)

        Returns:
            q_values: (B, num_actions)
            new_hidden: (B, hidden_dim)
        """
        encoded = self.obs_encoder(obs)
        h = self.gru(encoded, hidden)
        q_values = self.q_head(h)
        return q_values, h

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


def dqn_v1_generate_episode(
    network: DQNv1Network,
    devices,
    layers,
    tensor_size: float,
    num_layers: int,
    num_devices: int,
    device: torch.device,
    epsilon: float = 0.0,
    temperature: float = 1.0,
):
    """Generate one episode using DQN with epsilon-greedy or temperature sampling.

    Returns:
        transitions: list of (obs, action, reward, next_obs, done, log_info)
        ordering: final device ordering
        partition: final partition
        tpot: TPOT of the partition
    """
    network.eval()
    base_obs = build_observation(devices, layers, tensor_size, num_layers, num_devices)

    selected = []
    selected_mask = np.zeros(MAX_DEVICES, dtype=np.float32)
    hidden = network.init_hidden(1, device)
    transitions = []
    episode_reward = 0.0

    for step in range(num_devices + 1):  # +1 for potential STOP
        seq_obs = build_sequential_observation(base_obs, selected_mask, step, num_devices)
        obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(device)

        # Store hidden state BEFORE this step's forward pass
        h_before = hidden.clone()

        with torch.no_grad():
            q_values, hidden = network(obs_tensor, hidden)

        # Build action mask
        mask = np.ones(network.num_actions, dtype=bool)
        for s in selected:
            mask[s] = False  # already selected -> invalid
        # STOP always valid
        mask[MAX_DEVICES] = True
        # Mask out devices beyond num_devices
        for d in range(num_devices, MAX_DEVICES):
            mask[d] = False

        # Epsilon-greedy or temperature sampling
        if np.random.random() < epsilon:
            valid_indices = np.where(mask)[0]
            action = np.random.choice(valid_indices)
        else:
            q_np = q_values.cpu().numpy().flatten()
            q_np[~mask] = -np.inf
            if temperature != 1.0 and temperature > 0:
                logits = q_np / temperature
                logits -= logits.max()
                probs = np.exp(logits)
                probs[~mask] = 0
                probs /= probs.sum()
                action = np.random.choice(len(probs), p=probs)
            else:
                action = int(np.argmax(q_np))

        is_stop = (action == MAX_DEVICES) or (step == num_devices)

        if is_stop and step > 0:
            # STOP: compute final reward
            if not selected:
                # Must select at least one device
                valid = [d for d in range(num_devices) if d not in selected]
                selected.append(valid[0])
            ordering = selected
            partition = min_max_bottleneck_dp(
                num_layers, ordering, devices, layers, tensor_size
            )
            tpot = compute_simple_tpot(partition, devices, layers, tensor_size)
            # Final transition
            next_mask = selected_mask.copy()
            next_obs = build_sequential_observation(base_obs, next_mask, step + 1, num_devices)
            transitions.append((seq_obs, action, 0.0, next_obs, True,
                                h_before.squeeze(0).cpu().numpy()))
            return transitions, ordering, partition, tpot

        if action < num_devices and action not in selected:
            selected.append(action)
            selected_mask[action] = 1.0

        # Intermediate step (no reward yet)
        next_obs = build_sequential_observation(base_obs, selected_mask, step + 1, num_devices)
        transitions.append((seq_obs, action, 0.0, next_obs, False,
                            h_before.squeeze(0).cpu().numpy()))

    # Fallback: if we went through all devices without STOP
    if not selected:
        selected = [0]
    ordering = selected
    partition = min_max_bottleneck_dp(num_layers, ordering, devices, layers, tensor_size)
    tpot = compute_simple_tpot(partition, devices, layers, tensor_size)
    return transitions, ordering, partition, tpot


def dqn_v1_inference(
    network: DQNv1Network,
    devices,
    layers,
    tensor_size: float,
    num_layers: int,
    num_devices: int,
    device: torch.device,
    num_candidates: int = 10,
    dp_tpot: float = None,
    beam_tpot: float = None,
):
    """Pure greedy inference — single deterministic pass."""
    _, ordering, partition, tpot = dqn_v1_generate_episode(
        network, devices, layers, tensor_size, num_layers, num_devices,
        device, epsilon=0.0
    )
    return partition, tpot
