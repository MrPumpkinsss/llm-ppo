"""
simulator.py
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class LayerProfile:
    flops: float
    activation_size: float
    param_size: float


@dataclass
class DeviceProfile:
    device_id: int
    compute_power: float
    memory: float


@dataclass
class ClusterConfig:
    devices: List[DeviceProfile]
    bandwidth_matrix: np.ndarray


def generate_llm_layers(num_layers: int = 28) -> List[LayerProfile]:
    layers = []
    for i in range(num_layers):
        pos = 1.0 + 0.4 * np.sin(2 * np.pi * i / num_layers)
        phase = i % 3
        if phase == 0:
            flops, param, act = 0.9 * pos, 0.40, 0.030
        elif phase == 1:
            flops, param, act = 0.35 * pos, 0.80, 0.016
        else:
            flops, param, act = 0.25 * pos, 0.30, 0.020
        layers.append(LayerProfile(flops=flops, activation_size=act, param_size=param))
    return layers


def generate_heterogeneous_cluster(num_devices: int, seed: int = 42) -> ClusterConfig:
    rng = np.random.RandomState(seed)
    archetypes = [
        ((80, 120), (2.5, 4.0)),
        ((40, 70),  (4.0, 6.0)),
        ((15, 35),  (6.0, 10.0)),
        ((60, 100), (3.5, 5.5)),
    ]
    devices = []
    for i in range(num_devices):
        cr, mr = archetypes[i % len(archetypes)]
        devices.append(DeviceProfile(i, rng.uniform(*cr), rng.uniform(*mr)))

    bw = np.zeros((num_devices, num_devices))
    for i in range(num_devices):
        for j in range(num_devices):
            if i == j:
                bw[i][j] = float('inf')
            else:
                b = rng.uniform(0.5, 2.0)
                if abs(i - j) == 1:
                    b *= rng.uniform(1.5, 3.0)
                bw[i][j] = b
    np.fill_diagonal(bw, float('inf'))
    return ClusterConfig(devices=devices, bandwidth_matrix=bw)


def simulate_inference_tpot(layers, cluster, partition, batch_size=1):
    n = len(layers)
    partition = [int(p) for p in partition]
    dev_times = {}
    for i in range(n):
        d = partition[i]
        t = (layers[i].flops * batch_size / cluster.devices[d].compute_power) * 1000.0
        dev_times[d] = dev_times.get(d, 0.0) + t

    comm_total = 0.0
    comms = []
    for i in range(n - 1):
        d1, d2 = partition[i], partition[i + 1]
        if d1 != d2:
            ct = (layers[i].activation_size * batch_size / cluster.bandwidth_matrix[d1][d2]) * 1000.0 + 0.3
            comm_total += ct
            comms.append({'from': d1, 'to': d2, 'layer': i, 'time_ms': ct})

    total_compute = sum(dev_times.values())
    return {
        'tpot': total_compute + comm_total,
        'total_compute_time': total_compute,
        'total_comm_time': comm_total,
        'max_device_compute': max(dev_times.values()) if dev_times else 0.0,
        'device_compute_times': dev_times,
        'comm_events': comms,
        'num_device_switches': len(comms),
    }


def check_memory_feasibility(layers, cluster, partition):
    nd = len(cluster.devices)
    usage = [0.0] * nd
    for i, d in enumerate(partition):
        usage[int(d)] += layers[i].param_size
    return all(usage[d] <= cluster.devices[d].memory for d in range(nd))


def verify_cluster_constraints(layers, cluster):
    tp = sum(l.param_size for l in layers)
    mm = max(d.memory for d in cluster.devices)
    tm = sum(d.memory for d in cluster.devices)
    return {'total_params': tp, 'max_single_mem': mm, 'total_cluster_mem': tm,
            'must_distribute': tp > mm, 'feasible': tp <= tm}