"""Baseline algorithms: Dynamic Programming and Greedy for layer partitioning."""
import numpy as np
from typing import Tuple, List
from environment import DeviceCluster, LayerModel, compute_simple_tpot


def greedy_partition(
    num_layers: int,
    num_devices: int,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> list:
    """Greedy algorithm: assign layers one by one to the device that minimizes
    incremental TPOT increase.

    Strategy: iterate through layers, for each new layer try adding it to each
    device and pick the one that gives the best (lowest) TPOT so far.
    """
    partition = [0] * num_layers  # Start with all on device 0

    # Precompute: assign first layer to best single device
    best_dev = 0
    best_time = float('inf')
    for d in range(num_devices):
        t = layers.compute_costs[0] / devices.compute_power[d]
        if t < best_time:
            best_time = t
            best_dev = d
    partition[0] = best_dev

    for layer_idx in range(1, num_layers):
        best_dev = -1
        best_tpot = float('inf')

        # Try each device (must respect continuous assignment for the
        # devices we've already used in order)
        max_dev_used = max(partition[:layer_idx])
        for d in range(max_dev_used + 1):
            # Only assign to d if continuous (d >= all previous assignments
            # that come after, and d <= max used)
            # Actually for greedy with continuous constraint:
            # layer i can only be on device d where d >= partition[layer_idx-1]
            # is NOT required; we just need the final partition to be non-decreasing.
            # Greedy approach: try all valid devices
            candidate = partition[:layer_idx] + [d] + [0] * (num_layers - layer_idx - 1)
            tpot = compute_simple_tpot(candidate, devices, layers, tensor_size)
            if tpot < best_tpot:
                best_tpot = tpot
                best_dev = d

        partition[layer_idx] = best_dev

    return partition


def greedy_partition_advanced(
    num_layers: int,
    num_devices: int,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> list:
    """Advanced greedy: try all (num_devices) split points for each stage.

    Uses the observation that with K devices, we need K-1 split points.
    Greedily find the split point that minimizes TPOT, one at a time.
    """
    if num_devices == 1:
        return [0] * num_layers

    # Start: all layers on device 0
    # We'll build split points one at a time
    boundaries = [0, num_layers]  # boundaries[0]=0, boundaries[-1]=num_layers
    dev_order = sorted(range(num_devices), key=lambda d: -devices.compute_power[d])

    # Assign the fastest device to handle the most layers initially
    # Then greedily split

    # Initial partition: assign all to best device
    best_dev_idx = dev_order[0]
    partition = [best_dev_idx] * num_layers

    # Now greedily add split points using remaining devices
    for stage_idx in range(1, num_devices):
        target_dev = dev_order[stage_idx]  # Original device ID
        best_split = -1
        best_tpot = float('inf')

        # Try every possible split point in every existing stage
        for seg in range(len(boundaries) - 1):
            start = boundaries[seg]
            end = boundaries[seg + 1]
            for split in range(start + 1, end):
                # Create partition with this split
                candidate = list(partition)
                for l in range(split, num_layers):
                    if candidate[l] == candidate[split - 1]:
                        candidate[l] = target_dev
                    else:
                        break
                tpot = compute_simple_tpot(candidate, devices, layers, tensor_size)
                if tpot < best_tpot:
                    best_tpot = tpot
                    best_split = split
                    best_seg = seg

        if best_split >= 0:
            # Update boundaries and partition
            boundaries.insert(best_seg + 1, best_split)
            # Reassign: everything from best_split onwards gets this device
            for l in range(best_split, num_layers):
                partition[l] = target_dev

    return partition


def dp_partition(
    num_layers: int,
    num_devices: int,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> list:
    """DP-optimal partition using compute-power sorted device order.

    Sorts devices by compute power (descending) before DP to get a better
    baseline than the fixed 0,1,2,... ordering.

    Time complexity: O(K * N^2) where K=num_devices, N=num_layers.
    """
    if num_devices == 1:
        return [0] * num_layers

    # Sort devices by compute power descending: order[i] is the actual device ID at pipeline position i
    order = sorted(range(num_devices), key=lambda d: devices.compute_power[d], reverse=True)

    # Run DP with this order, automatically choosing best number of devices
    # _dp_for_ordered_devices returns partition using actual device IDs from `order`
    return _dp_for_ordered_devices(
        num_layers, order, devices, layers, tensor_size
    )


def _dp_for_ordered_devices(
    num_layers: int,
    ordered_devices: list,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> list:
    """DP partition given a fixed device ordering (0-indexed in ordered_devices).

    Automatically selects the best number of devices to use (1 to K),
    not necessarily all K. This accounts for cases where using fewer
    devices yields lower TPOT due to transfer overhead.
    """
    K = len(ordered_devices)
    N = num_layers
    INF = float('inf')

    cum_costs = np.zeros(N + 1)
    for i in range(N):
        cum_costs[i + 1] = cum_costs[i] + layers.compute_costs[i]

    def range_cost(start, end):
        return cum_costs[end] - cum_costs[start]

    dp_tpot = np.full((K + 1, N + 1), INF)
    dp_max_comp = np.full((K + 1, N + 1), INF)
    dp_total_transfer = np.full((K + 1, N + 1), INF)
    dp_parent = np.full((K + 1, N + 1), -1, dtype=int)

    dp_tpot[0][0] = 0.0
    dp_max_comp[0][0] = 0.0
    dp_total_transfer[0][0] = 0.0

    for k in range(1, K + 1):
        dev_idx = ordered_devices[k - 1]
        for i in range(1, N + 1):
            if k == 1:
                stage_comp = range_cost(0, i) / devices.compute_power[dev_idx]
                dp_tpot[1][i] = stage_comp
                dp_max_comp[1][i] = stage_comp
                dp_total_transfer[1][i] = 0.0
                dp_parent[1][i] = 0
            else:
                prev_dev_idx = ordered_devices[k - 2]
                for j in range(k - 1, i):
                    stage_comp = range_cost(j, i) / devices.compute_power[dev_idx]
                    new_max_comp = max(dp_max_comp[k - 1][j], stage_comp)
                    new_transfer = dp_total_transfer[k - 1][j]
                    if j < i:
                        new_transfer += devices.transfer_time(prev_dev_idx, dev_idx, tensor_size)
                    new_tpot = new_max_comp + new_transfer

                    if new_tpot < dp_tpot[k][i]:
                        dp_tpot[k][i] = new_tpot
                        dp_max_comp[k][i] = new_max_comp
                        dp_total_transfer[k][i] = new_transfer
                        dp_parent[k][i] = j

    # Find best k to use by computing "stop here" TPOT for each k
    # Stopping at k means: take the best partition for (k, N) but extend
    # device k-1 to cover all remaining layers instead of stopping at the
    # original split point.
    best_k = K
    best_tpot = INF
    best_splits = {}  # k -> split point where device k-1 starts

    for k in range(1, K + 1):
        # Backtrack to find split points for partition (k, N)
        splits = {}
        remaining = N
        for step in range(k, 0, -1):
            split = dp_parent[step][remaining]
            splits[step] = split
            remaining = split

        # Compute TPOT when we stop at device k-1 (extend it to cover all layers)
        # Device k-1 starts at splits[k] and now covers layers [splits[k], N)
        last_dev = ordered_devices[k - 1]
        last_compute = range_cost(splits[k], N) / devices.compute_power[last_dev]
        max_compute = max(dp_max_comp[k - 1][splits[k]], last_compute)
        total_transfer = dp_total_transfer[k - 1][splits[k]]
        stop_tpot = max_compute + total_transfer

        if stop_tpot < best_tpot:
            best_tpot = stop_tpot
            best_k = k
            best_splits = splits

    # Build partition using best_k devices
    # Extend device best_k-1 to cover all remaining layers
    partition = [ordered_devices[best_k - 1]] * N  # default: all to last device
    remaining = best_splits[best_k]
    for k in range(best_k - 1, 0, -1):
        split = best_splits[k]
        dev_idx = ordered_devices[k - 1]
        for i in range(split, remaining):
            partition[i] = dev_idx
        remaining = split
    # k=1: device 0 covers [0, splits[1])
    for i in range(0, remaining):
        partition[i] = ordered_devices[0]

    return partition


def dp_for_device_order(
    num_layers: int,
    device_order: list,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> list:
    """DP partition given a fixed device ordering.

    Automatically selects the best number of devices to use (not necessarily all).
    This is used by PPO-v2 where the RL agent outputs the device order.
    """
    return _dp_for_ordered_devices(
        num_layers, device_order, devices, layers, tensor_size
    )
