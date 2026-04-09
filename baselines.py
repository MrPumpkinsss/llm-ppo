"""Baseline algorithms: Dynamic Programming, Greedy, Beam Search for layer partitioning."""
import itertools
import numpy as np
from typing import Tuple, List, Optional
from environment import DeviceCluster, LayerModel, compute_simple_tpot


def greedy_partition(
    num_layers: int,
    num_devices: int,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> list:
    """Greedy algorithm: assign layers one by one to minimize incremental TPOT."""
    partition = [0] * num_layers
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
        max_dev_used = max(partition[:layer_idx])
        for d in range(max_dev_used + 1):
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
    """Advanced greedy: iteratively find best split points."""
    if num_devices == 1:
        return [0] * num_layers
    boundaries = [0, num_layers]
    dev_order = sorted(range(num_devices), key=lambda d: -devices.compute_power[d])
    best_dev_idx = dev_order[0]
    partition = [best_dev_idx] * num_layers

    for stage_idx in range(1, num_devices):
        target_dev = dev_order[stage_idx]
        best_split = -1
        best_tpot = float('inf')
        best_seg = 0
        for seg in range(len(boundaries) - 1):
            start = boundaries[seg]
            end = boundaries[seg + 1]
            for split in range(start + 1, end):
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
            boundaries.insert(best_seg + 1, best_split)
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
    """DP-optimal partition using compute-power sorted device order."""
    if num_devices == 1:
        return [0] * num_layers
    order = sorted(range(num_devices), key=lambda d: devices.compute_power[d], reverse=True)
    return _dp_for_ordered_devices(num_layers, order, devices, layers, tensor_size)


def _dp_for_ordered_devices(
    num_layers: int,
    ordered_devices: list,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> list:
    """DP partition given a fixed device ordering. Auto-selects best k devices."""
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

    best_k = K
    best_tpot = INF
    best_splits = {}

    for k in range(1, K + 1):
        splits = {}
        remaining = N
        for step in range(k, 0, -1):
            split = dp_parent[step][remaining]
            splits[step] = split
            remaining = split

        last_dev = ordered_devices[k - 1]
        last_compute = range_cost(splits[k], N) / devices.compute_power[last_dev]
        max_compute = max(dp_max_comp[k - 1][splits[k]], last_compute)
        total_transfer = dp_total_transfer[k - 1][splits[k]]
        stop_tpot = max_compute + total_transfer

        if stop_tpot < best_tpot:
            best_tpot = stop_tpot
            best_k = k
            best_splits = splits

    partition = [ordered_devices[best_k - 1]] * N
    remaining = best_splits[best_k]
    for k in range(best_k - 1, 0, -1):
        split = best_splits[k]
        dev_idx = ordered_devices[k - 1]
        for i in range(split, remaining):
            partition[i] = dev_idx
        remaining = split
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
    """DP partition given a fixed device ordering."""
    return _dp_for_ordered_devices(num_layers, device_order, devices, layers, tensor_size)


def dp_partition_raw_order(
    num_layers: int,
    num_devices: int,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> list:
    """DP partition using raw device order (0,1,2,...) without sorting by compute power."""
    if num_devices == 1:
        return [0] * num_layers
    return _dp_for_ordered_devices(num_layers, list(range(num_devices)), devices, layers, tensor_size)


# ==================== New baselines ====================

def min_max_bottleneck_dp(
    num_layers: int,
    ordered_devices: list,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> list:
    """Min-max bottleneck DP: minimizes max per-stage compute time.

    Auto-selects the best number of devices to use.
    """
    K = len(ordered_devices)
    N = num_layers
    if K == 0:
        return [0] * N
    if N == 0:
        return []

    INF = float('inf')

    cum_costs = np.zeros(N + 1)
    for i in range(N):
        cum_costs[i + 1] = cum_costs[i] + layers.compute_costs[i]

    def range_cost(start, end):
        return cum_costs[end] - cum_costs[start]

    dp_bottleneck = np.full((K + 1, N + 1), INF)
    dp_transfer = np.full((K + 1, N + 1), INF)
    dp_parent = np.full((K + 1, N + 1), -1, dtype=int)

    dp_bottleneck[0][0] = 0.0
    dp_transfer[0][0] = 0.0

    for k in range(1, K + 1):
        dev_idx = ordered_devices[k - 1]
        power = devices.compute_power[dev_idx]
        for i in range(k, N + 1):
            if k == 1:
                stage_comp = range_cost(0, i) / power
                dp_bottleneck[1][i] = stage_comp
                dp_transfer[1][i] = 0.0
                dp_parent[1][i] = 0
            else:
                prev_dev_idx = ordered_devices[k - 2]
                for j in range(k - 1, i):
                    stage_comp = range_cost(j, i) / power
                    new_bottleneck = max(dp_bottleneck[k - 1][j], stage_comp)
                    new_transfer = dp_transfer[k - 1][j]
                    if j < i:
                        new_transfer += devices.transfer_time(prev_dev_idx, dev_idx, tensor_size)
                    new_tpot = new_bottleneck + new_transfer
                    cur_tpot = dp_bottleneck[k][i] + dp_transfer[k][i]
                    if new_tpot < cur_tpot:
                        dp_bottleneck[k][i] = new_bottleneck
                        dp_transfer[k][i] = new_transfer
                        dp_parent[k][i] = j

    # Find best k (auto-select number of devices)
    best_k = K
    best_tpot = INF
    best_splits = {}

    for k in range(1, K + 1):
        splits = {}
        remaining = N
        for step in range(k, 0, -1):
            split = int(dp_parent[step][remaining])
            splits[step] = split
            remaining = split

        last_dev = ordered_devices[k - 1]
        last_compute = range_cost(splits[k], N) / devices.compute_power[last_dev]
        bottleneck = max(float(dp_bottleneck[k - 1][splits[k]]), last_compute)
        transfer = float(dp_transfer[k - 1][splits[k]])
        stop_tpot = bottleneck + transfer

        if stop_tpot < best_tpot:
            best_tpot = stop_tpot
            best_k = k
            best_splits = splits

    # Build partition
    partition = [ordered_devices[best_k - 1]] * N
    remaining = best_splits[best_k]
    for k in range(best_k - 1, 0, -1):
        split = best_splits[k]
        dev_idx = ordered_devices[k - 1]
        for i in range(split, remaining):
            partition[i] = dev_idx
        remaining = split
    for i in range(0, remaining):
        partition[i] = ordered_devices[0]

    return partition


def beam_search_dp(
    num_layers: int,
    num_devices: int,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
    beam_width: int = 100,
) -> list:
    """Beam Search over device orderings + min-max bottleneck DP.

    Each beam state = partial device ordering. At each step, expand each beam
    by adding each remaining device, run DP, keep top-K by TPOT.
    """
    if num_devices == 1:
        return [0] * num_layers

    bw = beam_width  # use different name to avoid any shadowing

    # (tpot, ordering_tuple) pairs - store TPOT separately to avoid list mutation
    best_overall_tpot = float('inf')
    best_overall_partition = None

    # Initialize beam with single-device orderings
    beam_tpot = []
    beam_orders = []
    beam_parts = []

    for d in range(num_devices):
        part = min_max_bottleneck_dp(num_layers, [d], devices, layers, tensor_size)
        tpot = compute_simple_tpot(part, devices, layers, tensor_size)
        beam_tpot.append(tpot)
        beam_orders.append((d,))
        beam_parts.append(part)

        if tpot < best_overall_tpot:
            best_overall_tpot = tpot
            best_overall_partition = list(part)

    # Sort and prune beam
    indices = sorted(range(len(beam_tpot)), key=lambda i: beam_tpot[i])
    beam_tpot = [beam_tpot[i] for i in indices[:bw]]
    beam_orders = [beam_orders[i] for i in indices[:bw]]
    beam_parts = [beam_parts[i] for i in indices[:bw]]

    # Expand beam step by step
    for step in range(2, num_devices + 1):
        new_tpot = []
        new_orders = []
        new_parts = []

        for b in range(len(beam_orders)):
            ordering = list(beam_orders[b])
            used = set(ordering)
            for d in range(num_devices):
                if d in used:
                    continue
                new_ordering = ordering + [d]
                part = min_max_bottleneck_dp(
                    num_layers, new_ordering, devices, layers, tensor_size
                )
                tpot = compute_simple_tpot(part, devices, layers, tensor_size)

                new_tpot.append(tpot)
                new_orders.append(tuple(new_ordering))
                new_parts.append(part)

                if tpot < best_overall_tpot:
                    best_overall_tpot = tpot
                    best_overall_partition = list(part)

        if not new_tpot:
            break

        # Sort and prune
        indices = sorted(range(len(new_tpot)), key=lambda i: new_tpot[i])
        beam_tpot = [new_tpot[i] for i in indices[:bw]]
        beam_orders = [new_orders[i] for i in indices[:bw]]
        beam_parts = [new_parts[i] for i in indices[:bw]]

    return best_overall_partition


def brute_force_optimal(
    num_layers: int,
    num_devices: int,
    devices: DeviceCluster,
    layers: LayerModel,
    tensor_size: float = 1.0,
) -> Tuple[list, float]:
    """Brute-force: enumerate ALL device permutations, run DP on each, return globally optimal partition.

    Only practical for small num_devices (e.g. 4! = 24, 5! = 120, 6! = 720).
    Returns (best_partition, best_tpot).
    """
    best_partition = None
    best_tpot = float('inf')

    for perm in itertools.permutations(range(num_devices)):
        part = min_max_bottleneck_dp(num_layers, list(perm), devices, layers, tensor_size)
        tpot = compute_simple_tpot(part, devices, layers, tensor_size)
        if tpot < best_tpot:
            best_tpot = tpot
            best_partition = part

    return best_partition, best_tpot
