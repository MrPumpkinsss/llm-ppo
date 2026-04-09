"""Heterogeneous edge device simulation environment for LLM layer partitioning.

Simulates devices with different compute power, inter-device bandwidth,
and per-layer compute costs. TPOT (Time to Process One Token) is computed
as the key performance metric.
"""
import numpy as np
from typing import Optional, Tuple
from config import EnvConfig


class DeviceCluster:
    """Simulates a cluster of heterogeneous edge devices."""

    def __init__(self, num_devices: int, seed: Optional[int] = None):
        rng = np.random.RandomState(seed)
        # Compute power: FLOPS normalized to [0, 1] (higher is faster)
        self.compute_power = rng.uniform(0.1, 1.0, num_devices)
        # Inter-device bandwidth matrix (GB/s), symmetric
        # Diagonal is inf (same device, no transfer)
        self.bandwidth = rng.uniform(0.5, 5.0, (num_devices, num_devices))
        np.fill_diagonal(self.bandwidth, np.inf)
        # Make symmetric
        self.bandwidth = (self.bandwidth + self.bandwidth.T) / 2
        self.num_devices = num_devices

    def transfer_time(self, from_dev: int, to_dev: int, data_size: float) -> float:
        """Compute data transfer time between two devices."""
        if from_dev == to_dev:
            return 0.0
        return data_size / self.bandwidth[from_dev, to_dev]


class LayerModel:
    """Simulates the compute cost of transformer layers."""

    def __init__(self, num_layers: int, seed: Optional[int] = None):
        rng = np.random.RandomState(seed)
        # Relative compute cost per layer (normalized)
        # Realistic: attention layers tend to be heavier
        base_costs = rng.uniform(0.5, 2.0, num_layers)
        # Add some structure: even layers (attention) slightly heavier
        is_attention = np.arange(num_layers) % 2 == 0
        base_costs[is_attention] *= 1.3
        self.compute_costs = base_costs
        self.num_layers = num_layers


def compute_simple_tpot(
    partition: list,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> float:
    """Compute simple TPOT = sum(device compute times) + inter-device transfer times.

    For single-token autoregressive decode, layers are strictly sequential,
    so all stage compute times are additive (not parallel).

    Args:
        partition: list of device indices for each layer (continuous assignment).
                   e.g. [0,0,0,1,1,2,2,2] means layers 0-2 on device 0, 3-4 on device 1, etc.
        devices: DeviceCluster instance
        layers: LayerModel instance
        tensor_size: size of activation tensor at layer boundaries (GB)

    Returns:
        TPOT in arbitrary time units (lower is better)
    """
    num_layers = len(partition)
    assert len(partition) == layers.num_layers

    # Compute time per device
    device_compute = {}
    for layer_idx, dev_id in enumerate(partition):
        cost = layers.compute_costs[layer_idx] / devices.compute_power[dev_id]
        device_compute[dev_id] = device_compute.get(dev_id, 0.0) + cost

    if not device_compute:
        return float('inf')

    total_compute = sum(device_compute.values())

    # Transfer time: every boundary between different devices incurs transfer
    total_transfer = 0.0
    for i in range(num_layers - 1):
        if partition[i] != partition[i + 1]:
            total_transfer += devices.transfer_time(
                partition[i], partition[i + 1], tensor_size
            )

    return total_compute + total_transfer


def compute_pipeline_tpot(
    partition: list,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
    num_microbatches: int = 4,
) -> dict:
    """Compute pipeline-parallel TPOT considering bubble overhead.

    In pipeline parallelism, micro-batches are pipelined across stages.
    The effective throughput depends on the number of micro-batches and
    the imbalance between stages.

    Args:
        partition: list of device indices per layer
        devices: DeviceCluster instance
        layers: LayerModel instance
        tensor_size: activation tensor size (GB)
        num_microbatches: number of micro-batches for pipeline

    Returns:
        dict with keys: 'tpot', 'max_stage_time', 'bubble_fraction',
                        'stage_times', 'transfer_overhead'
    """
    num_layers = len(partition)
    assert len(partition) == layers.num_layers

    # Extract stages (contiguous groups on same device)
    stages = []
    current_dev = partition[0]
    current_layers = [0]
    for i in range(1, num_layers):
        if partition[i] == current_dev:
            current_layers.append(i)
        else:
            stages.append((current_dev, current_layers[:]))
            current_dev = partition[i]
            current_layers = [i]
    stages.append((current_dev, current_layers[:]))

    # Compute stage times (sum of layer costs / device power)
    stage_times = []
    stage_devs = []
    for dev_id, layer_indices in stages:
        total_cost = sum(layers.compute_costs[l] for l in layer_indices)
        compute_time = total_cost / devices.compute_power[dev_id]
        stage_times.append(compute_time)
        stage_devs.append(dev_id)

    if not stage_times:
        return {'tpot': float('inf'), 'max_stage_time': 0, 'bubble_fraction': 1.0,
                'stage_times': [], 'transfer_overhead': 0}

    max_stage_time = max(stage_times)
    num_stages = len(stages)

    # Transfer overhead per micro-batch between adjacent stages
    transfer_per_mb = 0.0
    for i in range(num_stages - 1):
        from_dev = stage_devs[i]
        to_dev = stage_devs[i + 1]
        transfer_per_mb += devices.transfer_time(from_dev, to_dev, tensor_size)

    # Pipeline execution time = startup + (num_microbatches - 1) * max_stage_time
    # startup = sum of all stage times (filling the pipeline)
    # Total tokens processed = num_microbatches
    # Effective TPOT = total_time / num_microbatches

    startup_time = sum(stage_times) + transfer_per_mb * num_stages
    total_pipeline_time = startup_time + (num_microbatches - 1) * (max_stage_time + transfer_per_mb)
    effective_tpot = total_pipeline_time / num_microbatches

    # Bubble fraction: time stages are idle vs total time
    ideal_time = num_microbatches * max_stage_time
    bubble = total_pipeline_time - ideal_time
    bubble_fraction = max(0, bubble / total_pipeline_time) if total_pipeline_time > 0 else 1.0

    return {
        'tpot': effective_tpot,
        'max_stage_time': max_stage_time,
        'bubble_fraction': bubble_fraction,
        'stage_times': stage_times,
        'stage_devs': stage_devs,
        'transfer_overhead': transfer_per_mb,
        'num_stages': num_stages,
        'total_pipeline_time': total_pipeline_time,
    }


def create_random_config(
    num_layers: int, num_devices: int, seed: Optional[int] = None
) -> Tuple[DeviceCluster, LayerModel, int]:
    """Create a random device cluster and layer model configuration."""
    devices = DeviceCluster(num_devices, seed=seed)
    layers = LayerModel(num_layers, seed=seed + 10000 if seed is not None else None)
    tensor_size = 1.0  # Default 1GB activation
    return devices, layers, tensor_size
