"""
env.py - 切割点式动作空间

核心思路：
agent输出 max_devices-1 个切割比例 + max_devices 个设备排列
= 把28层切成若干连续段，每段分给一个设备

动作空间: Box(0,1, shape=(2*max_devices-1,))
  前 max_devices-1 维: 切割比例(经softmax归一化)
  后 max_devices 维: 设备优先级(经argsort得到设备排列)
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


class LLMPartitionEnv(gym.Env):
    """单步环境，连续动作 → 连续块分配"""

    def __init__(self, min_devices=2, max_devices=MAX_DEVICES, seed=None):
        super().__init__()
        self.num_layers = NUM_LAYERS
        self.min_devices = min_devices
        self.max_devices = max_devices

        # 动作: [cut_ratios(max_devices-1) + device_priorities(max_devices)]
        act_dim = 2 * max_devices - 1
        self.action_space = spaces.Box(-1.0, 1.0, (act_dim,), np.float32)

        # 观察: 设备特征 + 层统计 + 带宽 + 全局
        dev_feat = max_devices * 3   # compute, memory, rank
        layer_stat = 6               # 层统计量(均值/方差 of flops, param, act)
        bw_feat = max_devices * max_devices
        global_feat = 3
        self.obs_size = dev_feat + layer_stat + bw_feat + global_feat
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_size,), np.float32)

        self._rng = np.random.RandomState(seed)
        self.layers = None
        self.cluster = None
        self.num_devices = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.num_devices = self._rng.randint(self.min_devices, self.max_devices + 1)
        for _ in range(50):
            cs = self._rng.randint(0, 100000)
            self.cluster = generate_heterogeneous_cluster(self.num_devices, seed=cs)
            self.layers = generate_llm_layers(self.num_layers)
            c = verify_cluster_constraints(self.layers, self.cluster)
            if c['must_distribute'] and c['feasible']:
                break
        return self._get_obs(), {'num_devices': self.num_devices}

    def _get_obs(self):
        obs = []
        MC, MM = 150.0, 12.0
        pows = []
        for d in range(self.max_devices):
            if d < self.num_devices:
                dev = self.cluster.devices[d]
                pows.append(dev.compute_power)
                obs += [min(1, dev.compute_power / MC), min(1, dev.memory / MM), 0.0]
            else:
                pows.append(0)
                obs += [0, 0, 0]
        # 填排名
        for rank, idx in enumerate(np.argsort(pows)[::-1]):
            obs[idx * 3 + 2] = 1.0 - rank / max(1, self.max_devices - 1)

        # 层统计
        flops_arr = [l.flops for l in self.layers]
        param_arr = [l.param_size for l in self.layers]
        act_arr = [l.activation_size for l in self.layers]
        obs += [
            np.mean(flops_arr) / 2.0, np.std(flops_arr) / 1.0,
            np.mean(param_arr) / 1.5, np.std(param_arr) / 0.5,
            np.mean(act_arr) / 0.1, np.std(act_arr) / 0.05,
        ]

        # 带宽
        for i in range(self.max_devices):
            for j in range(self.max_devices):
                if i == j or i >= self.num_devices or j >= self.num_devices:
                    obs.append(0)
                else:
                    obs.append(min(1, self.cluster.bandwidth_matrix[i][j] / 10))

        # 全局
        tp = sum(l.param_size for l in self.layers)
        tf = sum(l.flops for l in self.layers)
        obs += [self.num_devices / self.max_devices, min(1, tp / 20), min(1, tf / 30)]

        return np.clip(np.array(obs, np.float32), 0, 1)

    def _action_to_partition(self, action):
        """把连续动作转换成连续块分配"""
        nd = self.num_devices
        action = np.array(action, dtype=np.float64)

        # 1. 解析切割比例: 前 max_devices-1 维
        cut_raw = action[:self.max_devices - 1]
        # 只用前 nd-1 个切割点 (nd个设备需要nd-1个切割)
        cut_raw = cut_raw[:nd - 1]
        # sigmoid + 归一化 → 每段的比例
        cut_probs = 1.0 / (1.0 + np.exp(-cut_raw * 2))  # 温和sigmoid
        # 加上最后一段
        all_probs = np.concatenate([cut_probs, [1.0]])
        all_probs = all_probs / (all_probs.sum() + 1e-8)

        # 2. 转换为层数
        seg_sizes_float = all_probs * self.num_layers
        seg_sizes = np.round(seg_sizes_float).astype(int)
        seg_sizes = np.maximum(seg_sizes, 1)  # 每段至少1层

        # 调整总和
        diff = self.num_layers - seg_sizes.sum()
        if diff > 0:
            # 需要增加层: 给比例最大的段增加
            for _ in range(diff):
                idx = np.argmax(all_probs - seg_sizes / self.num_layers)
                seg_sizes[idx] += 1
        elif diff < 0:
            # 需要减少层: 从最大的段减少
            for _ in range(-diff):
                idx = np.argmax(seg_sizes)
                if seg_sizes[idx] > 1:
                    seg_sizes[idx] -= 1

        # 3. 解析设备排列: 后 max_devices 维
        dev_raw = action[self.max_devices - 1:]
        # 只用前nd个
        dev_priorities = dev_raw[:nd]
        # argsort → 设备排列顺序
        dev_order = np.argsort(-dev_priorities)  # 高优先 → 前面

        # 4. 构造partition
        partition = []
        for seg_idx in range(nd):
            dev_id = int(dev_order[seg_idx])
            partition += [dev_id] * int(seg_sizes[seg_idx])

        # 安全截断/补齐
        partition = partition[:self.num_layers]
        while len(partition) < self.num_layers:
            partition.append(partition[-1])

        # 5. 内存修复
        partition = self._fix_memory(partition)

        return partition

    def _fix_memory(self, partition):
        """修复内存违规：把超内存设备的层移到相邻段的设备"""
        partition = list(partition)
        nd = self.num_devices
        mem = [0.0] * nd
        for i, d in enumerate(partition):
            mem[d] += self.layers[i].param_size

        for d in range(nd):
            while mem[d] > self.cluster.devices[d].memory + 1e-6:
                # 从该设备的最后一层开始移走
                moved = False
                for i in range(self.num_layers - 1, -1, -1):
                    if partition[i] == d:
                        # 找有空间的设备
                        best, best_r = None, -1
                        for ad in range(nd):
                            if ad == d: continue
                            r = self.cluster.devices[ad].memory - mem[ad]
                            if r >= self.layers[i].param_size and r > best_r:
                                best, best_r = ad, r
                        if best is not None:
                            mem[d] -= self.layers[i].param_size
                            mem[best] += self.layers[i].param_size
                            partition[i] = best
                            moved = True
                            break
                if not moved:
                    break
        return partition

    def step(self, action):
        partition = self._action_to_partition(action)

        result = simulate_inference_tpot(self.layers, self.cluster, partition)
        feasible = check_memory_feasibility(self.layers, self.cluster, partition)

        if not feasible:
            reward = -100.0
        else:
            reward = -result['tpot'] / 10.0

        info = {
            'tpot': result['tpot'], 'partition': partition, 'feasible': feasible,
            'total_compute_time': result['total_compute_time'],
            'total_comm_time': result['total_comm_time'],
            'num_device_switches': result['num_device_switches'],
            'max_device_compute': result['max_device_compute'],
            'device_compute_times': result['device_compute_times'],
            'comm_events': result['comm_events'],
        }
        return self._get_obs(), reward, True, False, info


class FixedClusterEnv(LLMPartitionEnv):
    def __init__(self, num_devices=4, cluster_seed=42, **kw):
        super().__init__(min_devices=num_devices, max_devices=MAX_DEVICES, **kw)
        self.fixed_nd = num_devices
        self.fixed_cs = cluster_seed

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.num_devices = self.fixed_nd
        self.cluster = generate_heterogeneous_cluster(self.num_devices, seed=self.fixed_cs)
        self.layers = generate_llm_layers(self.num_layers)
        return self._get_obs(), {'num_devices': self.num_devices}