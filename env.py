"""
env.py - 逐层决策，密集reward shaping
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple
from simulator import (
    generate_llm_layers, generate_heterogeneous_cluster,
    simulate_inference_tpot, check_memory_feasibility,
    verify_cluster_constraints,
)

NUM_LAYERS = 28
MAX_DEVICES = 6


def _compute_greedy_mem_partition(layers, cluster):
    """内存感知贪心基线，作为reward参考"""
    nd = len(cluster.devices)
    part, dm = [], [0.0] * nd
    for li in range(len(layers)):
        best_d, best_s = None, float('inf')
        for d in range(nd):
            if dm[d] + layers[li].param_size > cluster.devices[d].memory:
                continue
            ct = (layers[li].flops / cluster.devices[d].compute_power) * 1000
            comm = 0
            if part and part[-1] != d:
                comm = (layers[li - 1].activation_size / cluster.bandwidth_matrix[part[-1]][d]) * 1000 + 0.3
            if ct + comm < best_s:
                best_s, best_d = ct + comm, d
        if best_d is None:
            best_d = max(range(nd), key=lambda d: cluster.devices[d].memory - dm[d])
        part.append(best_d)
        dm[best_d] += layers[li].param_size
    return part


class LLMPartitionEnv(gym.Env):
    """逐层分配，28步完成"""

    def __init__(self, min_devices=2, max_devices=MAX_DEVICES, seed=None):
        super().__init__()
        self.num_layers = NUM_LAYERS
        self.min_devices = min_devices
        self.max_devices = max_devices

        self.action_space = spaces.Discrete(max_devices)

        # 观察空间：
        # 每设备(9): 算力, 排名, 总内存, 剩余内存, 已分层数, 已分flops比, 已分param比, 累积计算时间, 有效标记
        # 全局(8): 层进度, 当前层flops, 当前层param, 当前层act, 已用设备比, 累积通信次数比, 剩余param比, 负载不均衡度
        # 上一层设备one-hot(max_devices)
        # 带宽特征(max_devices)
        # 每设备累积计算时间(max_devices)
        dev_f = max_devices * 9
        global_f = 8
        prev_f = max_devices
        bw_f = max_devices
        time_f = max_devices
        self.obs_size = dev_f + global_f + prev_f + bw_f + time_f
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_size,), np.float32)

        self._rng = np.random.RandomState(seed)
        self._reset_state()

    def _reset_state(self):
        self.layers = None
        self.cluster = None
        self.num_devices = 0
        self.current_layer = 0
        self.partition = []
        self.dev_mem_used = np.zeros(MAX_DEVICES)
        self.dev_flops = np.zeros(MAX_DEVICES)
        self.dev_params = np.zeros(MAX_DEVICES)
        self.dev_compute_times = np.zeros(MAX_DEVICES)
        self.total_flops = 0
        self.total_params = 0
        self.dev_rank = np.zeros(MAX_DEVICES)
        self.greedy_tpot = None
        self.cumulative_comm = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._reset_state()
        self.num_devices = self._rng.randint(self.min_devices, self.max_devices + 1)
        for _ in range(50):
            cs = self._rng.randint(0, 100000)
            self.cluster = generate_heterogeneous_cluster(self.num_devices, seed=cs)
            self.layers = generate_llm_layers(self.num_layers)
            c = verify_cluster_constraints(self.layers, self.cluster)
            if c['must_distribute'] and c['feasible']:
                break

        self.total_flops = sum(l.flops for l in self.layers)
        self.total_params = sum(l.param_size for l in self.layers)

        pows = [self.cluster.devices[d].compute_power if d < self.num_devices else 0
                for d in range(MAX_DEVICES)]
        for rank, idx in enumerate(np.argsort(pows)[::-1]):
            self.dev_rank[idx] = rank / max(1, MAX_DEVICES - 1)

        # 预计算贪心基线
        gp = _compute_greedy_mem_partition(self.layers, self.cluster)
        gr = simulate_inference_tpot(self.layers, self.cluster, gp)
        self.greedy_tpot = gr['tpot']

        return self._get_obs(), {'num_devices': self.num_devices, 'greedy_tpot': self.greedy_tpot}

    def _get_obs(self):
        obs = []
        MC, MM, MT = 150.0, 12.0, 50.0
        for d in range(MAX_DEVICES):
            if d < self.num_devices:
                dev = self.cluster.devices[d]
                obs += [
                    min(1, dev.compute_power / MC),
                    1.0 - self.dev_rank[d],
                    min(1, dev.memory / MM),
                    min(1, max(0, dev.memory - self.dev_mem_used[d]) / MM),
                    sum(1 for p in self.partition if p == d) / max(1, self.num_layers),
                    self.dev_flops[d] / max(0.001, self.total_flops),
                    self.dev_params[d] / max(0.001, self.total_params),
                    min(1, self.dev_compute_times[d] / MT),
                    1.0,
                ]
            else:
                obs += [0] * 9

        li = self.current_layer
        progress = li / self.num_layers
        if li < self.num_layers:
            l = self.layers[li]
            lf = min(1, l.flops / 2.0)
            lp = min(1, l.param_size / 1.5)
            la = min(1, l.activation_size / 0.1)
        else:
            lf = lp = la = 0

        used_dev = len(set(self.partition)) / max(1, self.num_devices) if self.partition else 0
        comm_count = sum(1 for i in range(1, len(self.partition)) if self.partition[i] != self.partition[i - 1])
        comm_ratio = comm_count / max(1, self.num_layers - 1)
        rem_params = sum(self.layers[i].param_size for i in range(li, self.num_layers)) / max(0.001, self.total_params)

        used_t = [self.dev_compute_times[d] for d in range(self.num_devices) if self.dev_compute_times[d] > 0]
        imbalance = min(1, np.std(used_t) / (np.mean(used_t) + 1e-8)) if len(used_t) > 1 else 0

        obs += [progress, lf, lp, la, used_dev, comm_ratio, min(1, rem_params), imbalance]

        prev_oh = np.zeros(MAX_DEVICES)
        if self.partition:
            prev_oh[self.partition[-1]] = 1.0
        obs += prev_oh.tolist()

        bw = np.zeros(MAX_DEVICES)
        if self.partition:
            pd = self.partition[-1]
            for d in range(self.num_devices):
                bw[d] = 1.0 if d == pd else min(1, self.cluster.bandwidth_matrix[pd][d] / 5.0)
        else:
            bw[:self.num_devices] = 1.0
        obs += bw.tolist()

        dt = np.zeros(MAX_DEVICES)
        for d in range(self.num_devices):
            dt[d] = min(1, self.dev_compute_times[d] / MT)
        obs += dt.tolist()

        return np.clip(np.array(obs, np.float32), 0, 1)

    def step(self, action):
        action = int(action)
        info = {}

        if action >= self.num_devices:
            action = action % self.num_devices

        layer = self.layers[self.current_layer]

        # 内存检查
        if self.dev_mem_used[action] + layer.param_size > self.cluster.devices[action].memory:
            candidates = [(self.cluster.devices[d].compute_power, d) for d in range(self.num_devices)
                          if self.dev_mem_used[d] + layer.param_size <= self.cluster.devices[d].memory]
            if candidates:
                action = max(candidates)[1]
                info['mem_redirect'] = True
            else:
                action = max(range(self.num_devices),
                             key=lambda d: self.cluster.devices[d].memory - self.dev_mem_used[d])
                info['forced'] = True

        # 执行分配
        compute_ms = (layer.flops / self.cluster.devices[action].compute_power) * 1000.0
        comm_ms = 0.0
        if self.partition and self.partition[-1] != action:
            prev_layer = self.layers[self.current_layer - 1]
            bw = self.cluster.bandwidth_matrix[self.partition[-1]][action]
            comm_ms = (prev_layer.activation_size / bw) * 1000.0 + 0.3

        self.partition.append(action)
        self.dev_mem_used[action] += layer.param_size
        self.dev_flops[action] += layer.flops
        self.dev_params[action] += layer.param_size
        self.dev_compute_times[action] += compute_ms
        self.cumulative_comm += comm_ms
        self.current_layer += 1

        terminated = self.current_layer >= self.num_layers

        if terminated:
            part = [int(p) for p in self.partition]
            result = simulate_inference_tpot(self.layers, self.cluster, part)
            tpot = result['tpot']
            feasible = check_memory_feasibility(self.layers, self.cluster, part)

            if not feasible:
                reward = -50.0
            else:
                # 相对贪心基线的改善: 正 = 比贪心好, 负 = 比贪心差
                improvement = (self.greedy_tpot - tpot) / self.greedy_tpot
                reward = improvement * 50.0  # 比贪心好10% → reward=+5

            info.update({
                'tpot': tpot, 'partition': part, 'feasible': feasible,
                'total_compute_time': result['total_compute_time'],
                'total_comm_time': result['total_comm_time'],
                'num_device_switches': result['num_device_switches'],
                'max_device_compute': result['max_device_compute'],
                'device_compute_times': result['device_compute_times'],
                'comm_events': result['comm_events'],
                'greedy_tpot': self.greedy_tpot,
            })
        else:
            # 密集中间reward：引导agent
            reward = 0.0

            # 1. 通信惩罚（按实际通信时间比例）
            if comm_ms > 0:
                reward -= comm_ms / self.greedy_tpot * 2.0

            # 2. 选了慢设备的惩罚（比最快设备慢多少）
            max_power = max(self.cluster.devices[d].compute_power for d in range(self.num_devices))
            speed_ratio = self.cluster.devices[action].compute_power / max_power
            if speed_ratio < 0.5:
                reward -= (1.0 - speed_ratio) * 0.5

            # 3. 内存使用效率：如果设备快满了还在往里塞，小惩罚
            mem_usage = self.dev_mem_used[action] / self.cluster.devices[action].memory
            if mem_usage > 0.9:
                reward -= 0.3

        obs = self._get_obs()
        return obs, reward, terminated, False, info


class FixedClusterEnv(LLMPartitionEnv):
    def __init__(self, num_devices=4, cluster_seed=42, **kw):
        super().__init__(min_devices=num_devices, max_devices=MAX_DEVICES, **kw)
        self.fixed_nd = num_devices
        self.fixed_cs = cluster_seed

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._reset_state()
        self.num_devices = self.fixed_nd
        self.cluster = generate_heterogeneous_cluster(self.num_devices, seed=self.fixed_cs)
        self.layers = generate_llm_layers(self.num_layers)
        self.total_flops = sum(l.flops for l in self.layers)
        self.total_params = sum(l.param_size for l in self.layers)

        pows = [self.cluster.devices[d].compute_power if d < self.num_devices else 0
                for d in range(MAX_DEVICES)]
        for rank, idx in enumerate(np.argsort(pows)[::-1]):
            self.dev_rank[idx] = rank / max(1, MAX_DEVICES - 1)

        gp = _compute_greedy_mem_partition(self.layers, self.cluster)
        gr = simulate_inference_tpot(self.layers, self.cluster, gp)
        self.greedy_tpot = gr['tpot']

        return self._get_obs(), {'num_devices': self.num_devices, 'greedy_tpot': self.greedy_tpot}