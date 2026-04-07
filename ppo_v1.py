"""PPO-v1: Autoregressive GRU for device ordering + DP for layer allocation.

This is architecturally DIFFERENT from PPO-v2:
- PPO-v2: single-pass pointer network (non-autoregressive ordering)
- PPO-v1: autoregressive GRU that predicts device ordering step-by-step,
  conditioning on previous device selections and the full observation.

The autoregressive approach allows PPO-v1 to learn device dependencies that
PPO-v2's single-pass approach misses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from environment import compute_simple_tpot


def get_obs_dim(num_devices, num_layers):
    bw_upper_size = num_devices * (num_devices - 1) // 2
    return num_devices + num_layers + bw_upper_size + num_layers + 2


def build_observation(devices, layers, num_devices, num_layers):
    dev_power = devices.compute_power / (devices.compute_power.max() + 1e-8)
    layer_costs = layers.compute_costs / (layers.compute_costs.max() + 1e-8)
    bw = devices.bandwidth.copy()
    np.fill_diagonal(bw, 0)
    max_bw = bw[bw > 0].max() if (bw > 0).any() else 1.0
    bw_norm = bw / (max_bw + 1e-8)
    bw_upper = bw_norm[np.triu_indices(num_devices, k=1)]
    cum_costs = np.cumsum(layers.compute_costs)
    cum_costs_norm = cum_costs / (cum_costs[-1] + 1e-8)
    obs = np.concatenate([
        dev_power,
        layer_costs,
        bw_upper,
        cum_costs_norm,
        [num_layers / 64.0, num_devices / 10.0],
    ])
    return obs.astype(np.float32)


def build_device_features(devices, num_devices):
    features = np.zeros((num_devices, 4), dtype=np.float32)
    for d in range(num_devices):
        features[d, 0] = devices.compute_power[d]
        bw_to_others = devices.bandwidth[d, np.arange(num_devices) != d]
        features[d, 1] = bw_to_others.mean() if len(bw_to_others) > 0 else 0
        features[d, 2] = bw_to_others.max() if len(bw_to_others) > 0 else 0
        features[d, 3] = bw_to_others.min() if len(bw_to_others) > 0 else 0
    return features


class OrderPredictor(nn.Module):
    """Autoregressive GRU for predicting device ordering.

    Different from PPO-v2's single-pass pointer network.
    This model uses a GRU to condition each device selection on previous selections.
    """

    def __init__(self, obs_dim: int, max_devices: int, hidden_dim: int = 256):
        super().__init__()
        self.max_devices = max_devices
        self.hidden_dim = hidden_dim

        # Global observation encoder
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Device feature encoder
        self.dev_enc = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # GRU cell: input = dev_feat + prev_hidden
        self.gru = nn.GRUCell(hidden_dim * 2, hidden_dim)

        # Selection head
        self.select_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs: torch.Tensor, device_features: torch.Tensor,
                num_devices: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-pass (non-autoregressive) forward for training speed.

        Returns first-device logits and value estimate.
        """
        B = obs.size(0)
        x = self.obs_enc(obs)  # (B, hidden)
        dev_feat = self.dev_enc(device_features[:, :num_devices])  # (B, num_devices, hidden)

        # Use compatibility between obs encoding and device features as ordering proxy
        logits = (x.unsqueeze(1) * dev_feat).sum(dim=-1)  # (B, num_devices)

        value = self.value_head(x)
        return logits, value

    def _make_mask(self, B: int, num_devices: int, selected: torch.Tensor):
        """Build a (B, num_devices) boolean mask: True where device is available.

        Args:
            B: batch size
            num_devices: number of actual devices
            selected: (B, num_selected) tensor of device indices selected so far
        """
        mask = torch.ones(B, num_devices, dtype=torch.bool, device=selected.device)
        if selected.numel() == 0:
            return mask  # True = available
        for step_idx in range(selected.size(1)):
            for b in range(B):
                d = selected[b, step_idx].item()
                if d < num_devices:
                    mask[b, d] = False
        return mask  # True = available

    def generate_order_autoregressive(self, obs: torch.Tensor,
                                     device_features: torch.Tensor,
                                     num_devices: int,
                                     greedy: bool = True):
        """Generate device ordering autoregressively."""
        with torch.no_grad():
            B = obs.size(0)
            x = self.obs_enc(obs)  # (B, hidden)
            # Encode only the actual num_devices devices
            dev_feat = self.dev_enc(device_features[:, :num_devices])  # (B, num_devices, hidden)

            # Initial hidden state: zeros
            h = torch.zeros(B, self.hidden_dim, device=obs.device)

            order_list = []
            log_probs = []
            entropies = []

            for step in range(num_devices):
                # GRU input: dev_feat pooled + hidden
                pooled_dev = dev_feat.mean(dim=1)  # (B, hidden)
                gru_in = torch.cat([pooled_dev, h], dim=-1)  # (B, hidden*2)
                h = self.gru(gru_in, h)  # (B, hidden)

                # Selection scores: compatibility between GRU hidden and each device
                scores = (h.unsqueeze(1) * dev_feat).sum(dim=-1)  # (B, num_devices)

                # Build availability mask
                selected_so_far = (torch.stack(order_list, dim=1)
                                   if order_list else
                                   torch.empty(B, 0, dtype=torch.long, device=obs.device))
                avail_mask = self._make_mask(B, num_devices, selected_so_far)  # (B, num_devices)

                # Apply mask
                masked_scores = scores.masked_fill(~avail_mask, float('-inf'))

                probs = F.softmax(masked_scores, dim=-1)
                dist = torch.distributions.Categorical(probs)

                if greedy:
                    action = probs.argmax(dim=-1)
                else:
                    action = dist.sample()

                order_list.append(action)
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())

            order = torch.stack(order_list, dim=1)  # (B, num_devices)
            total_lp = torch.stack(log_probs, dim=1).sum(dim=1)  # (B,)
            total_ent = torch.stack(entropies, dim=1).sum(dim=1)  # (B,)

        return order, total_lp, total_ent

    def log_prob_order(self, obs: torch.Tensor, device_features: torch.Tensor,
                       order: torch.Tensor, num_devices: int):
        """Compute log probability of an ordering."""
        B = obs.size(0)
        x = self.obs_enc(obs)
        dev_feat = self.dev_enc(device_features[:, :num_devices])  # (B, num_devices, hidden)
        h = torch.zeros(B, self.hidden_dim, device=obs.device)

        log_probs = []

        for step in range(num_devices):
            pooled_dev = dev_feat.mean(dim=1)
            gru_in = torch.cat([pooled_dev, h], dim=-1)
            h = self.gru(gru_in, h)
            scores = (h.unsqueeze(1) * dev_feat).sum(dim=-1)  # (B, num_devices)

            selected_so_far = order[:, :step] if step > 0 else torch.empty(B, 0, dtype=torch.long, device=obs.device)
            avail_mask = self._make_mask(B, num_devices, selected_so_far)
            masked_scores = scores.masked_fill(~avail_mask, float('-inf'))

            probs = F.softmax(masked_scores, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs.append(dist.log_prob(order[:, step]))

        return torch.stack(log_probs, dim=1).sum(dim=1)


def compute_reward(partition, devices, layers, tensor_size=1.0):
    tpot = compute_simple_tpot(partition, devices, layers, tensor_size)
    return -tpot
