"""Shared utilities: observation builders, constants, reward computation."""
import numpy as np
import torch
from environment import DeviceCluster, LayerModel, compute_simple_tpot

# Global constants
MAX_DEVICES = 10
MAX_LAYERS = 64


def get_obs_dim(max_devices: int = MAX_DEVICES, max_layers: int = MAX_LAYERS) -> int:
    """Get observation dimension for the base observation."""
    return max_devices + max_layers + max_devices * (max_devices - 1) // 2 + 2


def get_seq_obs_dim(max_devices: int = MAX_DEVICES, max_layers: int = MAX_LAYERS) -> int:
    """Get observation dimension for sequential (autoregressive) observations.
    Base obs + selected_mask + step_norm."""
    return get_obs_dim(max_devices, max_layers) + max_devices + 2


def build_observation(
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float,
    num_devices: int,
    num_layers: int,
) -> np.ndarray:
    """Build base observation vector (padded to fixed size).

    Returns:
        np.ndarray of shape (get_obs_dim(),) with:
        - device compute powers (MAX_DEVICES, zero-padded, normalized)
        - layer compute costs (MAX_LAYERS, zero-padded, normalized)
        - bandwidth upper triangle (MAX_DEVICES*(MAX_DEVICES-1)//2, zero-padded, normalized)
        - global info [num_layers/64, num_devices/16]
    """
    max_dev = MAX_DEVICES
    max_lay = MAX_LAYERS
    bw_size = max_dev * (max_dev - 1) // 2

    # Device powers
    dev_power = np.zeros(max_dev, dtype=np.float32)
    dev_power[:devices.num_devices] = devices.compute_power.copy()

    # Layer costs
    lay_costs = np.zeros(max_lay, dtype=np.float32)
    lay_costs[:layers.num_layers] = layers.compute_costs.copy()
    # Normalize by max cost
    max_cost = lay_costs[:num_layers].max() if num_layers > 0 else 1.0
    if max_cost > 0:
        lay_costs[:num_layers] /= max_cost

    # Bandwidth upper triangle
    bw_upper = np.zeros(bw_size, dtype=np.float32)
    actual_nd = min(num_devices, devices.num_devices)
    if actual_nd > 1:
        idx = 0
        for i in range(actual_nd):
            for j in range(i + 1, actual_nd):
                bw_upper[idx] = devices.bandwidth[i, j] / 5.0  # normalize by max bw
                idx += 1

    # Global info
    global_info = np.array([num_layers / max_lay, num_devices / max_dev], dtype=np.float32)

    return np.concatenate([dev_power, lay_costs, bw_upper, global_info])


def build_device_features(
    devices: DeviceCluster,
    num_devices: int,
) -> np.ndarray:
    """Build per-device feature matrix (padded).

    Returns:
        np.ndarray of shape (MAX_DEVICES, 4) with columns:
        - compute power (normalized)
        - avg bandwidth to other devices
        - min bandwidth
        - max bandwidth
    """
    features = np.zeros((MAX_DEVICES, 4), dtype=np.float32)
    for i in range(num_devices):
        features[i, 0] = devices.compute_power[i]
        if num_devices > 1:
            bws = [devices.bandwidth[i, j] for j in range(num_devices) if j != i]
            features[i, 1] = np.mean(bws) / 5.0
            features[i, 2] = np.min(bws) / 5.0
            features[i, 3] = np.max(bws) / 5.0
    return features


def build_sequential_observation(
    base_obs: np.ndarray,
    selected_mask: np.ndarray,
    current_step: int,
    num_devices: int,
) -> np.ndarray:
    """Build sequential (autoregressive) observation.

    Args:
        base_obs: base observation from build_observation
        selected_mask: (MAX_DEVICES,) binary mask, 1.0 = already selected
        current_step: current step index
        num_devices: total number of devices

    Returns:
        np.ndarray of shape (get_seq_obs_dim(),)
    """
    step_norm = np.array([
        current_step / MAX_DEVICES,
        (num_devices - np.sum(selected_mask[:num_devices])) / MAX_DEVICES
    ], dtype=np.float32)
    return np.concatenate([base_obs, selected_mask, step_norm])


def compute_reward(
    partition: list,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float,
) -> float:
    """Compute raw reward: negative TPOT only. No shaping bonuses."""
    tpot = compute_simple_tpot(partition, devices, layers, tensor_size)
    return -tpot
