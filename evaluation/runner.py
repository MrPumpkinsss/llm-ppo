"""Evaluation runner for all 6 RL versions + baselines."""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from environment import DeviceCluster, LayerModel, create_random_config, compute_simple_tpot, compute_pipeline_tpot
from baselines import dp_partition, dp_partition_raw_order, greedy_partition_advanced, beam_search_dp, min_sum_tpot_dp

from agents.dqn_v1 import dqn_v1_inference
from agents.ppo_v2 import ppo_v2_inference
from agents.ppo_v3 import ppo_v3_inference
from agents.ppo_v4 import ppo_v4_inference
from agents.maskable_ppo_v5 import maskable_v5_inference
from agents.gnn_ppo_v6 import ppo_v6_inference
from agents.gnn_ar_ppo_v7 import ppo_v7_inference


@dataclass
class TestResult:
    """Per-test-case results."""
    test_id: int
    num_layers: int
    num_devices: int
    seed: int
    # TPOTs
    v1_tpot: float = float('inf')
    v2_tpot: float = float('inf')
    v3_tpot: float = float('inf')
    v4_tpot: float = float('inf')
    v5_tpot: float = float('inf')
    v6_tpot: float = float('inf')
    v7_tpot: float = float('inf')
    beam_tpot: float = float('inf')
    dp_tpot: float = float('inf')
    dp_raw_tpot: float = float('inf')
    greedy_tpot: float = float('inf')
    # Partitions
    v1_partition: list = field(default_factory=list)
    v2_partition: list = field(default_factory=list)
    v3_partition: list = field(default_factory=list)
    v4_partition: list = field(default_factory=list)
    v5_partition: list = field(default_factory=list)
    v6_partition: list = field(default_factory=list)
    v7_partition: list = field(default_factory=list)
    beam_partition: list = field(default_factory=list)
    dp_partition: list = field(default_factory=list)
    dp_raw_partition: list = field(default_factory=list)
    greedy_partition: list = field(default_factory=list)
    # Inference time (ms)
    v1_time_ms: float = 0.0
    v2_time_ms: float = 0.0
    v3_time_ms: float = 0.0
    v4_time_ms: float = 0.0
    v5_time_ms: float = 0.0
    v6_time_ms: float = 0.0
    v7_time_ms: float = 0.0
    beam_time_ms: float = 0.0
    dp_time_ms: float = 0.0
    dp_raw_time_ms: float = 0.0


def run_evaluation(
    networks: dict,
    config,
    eval_config,
) -> tuple:
    """Run evaluation across all test configurations.

    Args:
        networks: dict mapping 'v1'-'v5' to (network, metrics) tuples
        config: TrainConfig
        eval_config: EvalConfig

    Returns:
        (results_list, stats_dict)
    """
    import torch
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    results = []
    test_id = 0
    num_candidates = eval_config.num_inference_candidates
    beam_width = eval_config.beam_width_eval

    for nl in eval_config.num_layers_to_test:
        for nd in eval_config.num_devices_to_test:
            for trial in range(3):  # 3 trials per config
                seed = 10000 + nl * 100 + nd * 10 + trial
                devices, layers, ts = create_random_config(nl, nd, seed=seed)

                result = TestResult(
                    test_id=test_id,
                    num_layers=nl,
                    num_devices=nd,
                    seed=seed,
                )

                # DP baseline (sorted by compute power)
                t0 = time.time()
                dp_part = dp_partition(nl, nd, devices, layers, ts)
                result.dp_tpot = compute_simple_tpot(dp_part, devices, layers, ts)
                result.dp_partition = dp_part
                result.dp_time_ms = (time.time() - t0) * 1000

                # DP baseline (raw device order 0,1,2,...)
                t0 = time.time()
                dp_raw_part = dp_partition_raw_order(nl, nd, devices, layers, ts)
                result.dp_raw_tpot = compute_simple_tpot(dp_raw_part, devices, layers, ts)
                result.dp_raw_partition = dp_raw_part
                result.dp_raw_time_ms = (time.time() - t0) * 1000

                # Greedy baseline
                greedy_part = greedy_partition_advanced(nl, nd, devices, layers, ts)
                result.greedy_tpot = compute_simple_tpot(greedy_part, devices, layers, ts)
                result.greedy_partition = greedy_part

                # Beam Search baseline
                t0 = time.time()
                beam_part = beam_search_dp(nl, nd, devices, layers, ts, beam_width=beam_width)
                result.beam_tpot = compute_simple_tpot(beam_part, devices, layers, ts)
                result.beam_partition = beam_part
                result.beam_time_ms = (time.time() - t0) * 1000

                # V1: DQN + DP
                if 'v1' in networks:
                    net = networks['v1'][0]
                    t0 = time.time()
                    part, tpot = dqn_v1_inference(
                        net, devices, layers, ts, nl, nd, device, num_candidates
                    )
                    result.v1_tpot = tpot
                    result.v1_partition = part
                    result.v1_time_ms = (time.time() - t0) * 1000

                # V2: PPO Binary + DP
                if 'v2' in networks:
                    net = networks['v2'][0]
                    t0 = time.time()
                    part, tpot = ppo_v2_inference(
                        net, devices, layers, ts, nl, nd, device, num_candidates
                    )
                    result.v2_tpot = tpot
                    result.v2_partition = part
                    result.v2_time_ms = (time.time() - t0) * 1000

                # V3: PPO-Clip One-Shot + DP
                if 'v3' in networks:
                    net = networks['v3'][0]
                    t0 = time.time()
                    part, tpot = ppo_v3_inference(
                        net, devices, layers, ts, nl, nd, device, num_candidates
                    )
                    result.v3_tpot = tpot
                    result.v3_partition = part
                    result.v3_time_ms = (time.time() - t0) * 1000

                # V4: PPO-Clip AutoReg + DP
                if 'v4' in networks:
                    net = networks['v4'][0]
                    t0 = time.time()
                    part, tpot = ppo_v4_inference(
                        net, devices, layers, ts, nl, nd, device, num_candidates
                    )
                    result.v4_tpot = tpot
                    result.v4_partition = part
                    result.v4_time_ms = (time.time() - t0) * 1000

                # V5: Maskable PPO-Clip + DP
                if 'v5' in networks:
                    net = networks['v5'][0]
                    t0 = time.time()
                    part, tpot = maskable_v5_inference(
                        net, devices, layers, ts, nl, nd, device, num_candidates
                    )
                    result.v5_tpot = tpot
                    result.v5_partition = part
                    result.v5_time_ms = (time.time() - t0) * 1000

                # V6: GNN-Based PPO + DP
                if 'v6' in networks:
                    net = networks['v6'][0]
                    t0 = time.time()
                    part, tpot = ppo_v6_inference(
                        net, devices, layers, ts, nl, nd, device, num_candidates
                    )
                    result.v6_tpot = tpot
                    result.v6_partition = part
                    result.v6_time_ms = (time.time() - t0) * 1000

                # V7: Autoregressive GNN-PPO + Positional Encoding + DP
                if 'v7' in networks:
                    net = networks['v7'][0]
                    t0 = time.time()
                    part, tpot = ppo_v7_inference(
                        net, devices, layers, ts, nl, nd, device, num_candidates
                    )
                    result.v7_tpot = tpot
                    result.v7_partition = part
                    result.v7_time_ms = (time.time() - t0) * 1000

                results.append(result)
                test_id += 1

                if test_id % 10 == 0:
                    print(f"  Evaluated {test_id} test cases...")

    # Compute aggregate stats
    eval_versions = getattr(eval_config, 'eval_versions', None) or ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    stats = _compute_stats(results, eval_versions)
    return results, stats


def _compute_stats(results: list, eval_versions=None) -> dict:
    """Compute aggregate statistics from test results."""
    if not results:
        return {}

    if eval_versions is None:
        eval_versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']

    stats = {}
    total = len(results)

    for ver in eval_versions + ['beam', 'dp', 'dp_raw', 'greedy']:
        tpots = [getattr(r, f'{ver}_tpot') for r in results if getattr(r, f'{ver}_tpot') < float('inf')]
        if not tpots:
            continue

        dp_tpots = [r.dp_tpot for r in results if getattr(r, f'{ver}_tpot') < float('inf')]
        beam_tpots = [r.beam_tpot for r in results if getattr(r, f'{ver}_tpot') < float('inf')]

        gaps_dp = [(t - d) / d * 100 for t, d in zip(tpots, dp_tpots) if d > 0]
        gaps_beam = [(t - b) / b * 100 for t, b in zip(tpots, beam_tpots) if b > 0]
        wins_dp = sum(1 for t, d in zip(tpots, dp_tpots) if t <= d * 1.02)
        wins_beam = sum(1 for t, b in zip(tpots, beam_tpots) if t <= b * 1.02)

        stats[f'{ver}_avg_tpot'] = np.mean(tpots)
        stats[f'{ver}_avg_gap_to_dp'] = np.mean(gaps_dp) if gaps_dp else float('inf')
        stats[f'{ver}_avg_gap_to_beam'] = np.mean(gaps_beam) if gaps_beam else float('inf')
        stats[f'{ver}_wins_vs_dp'] = wins_dp
        stats[f'{ver}_wins_vs_beam'] = wins_beam
        stats[f'{ver}_win_rate_dp'] = wins_dp / len(tpots) * 100
        stats[f'{ver}_win_rate_beam'] = wins_beam / len(tpots) * 100

        # Avg inference time
        times = [getattr(r, f'{ver}_time_ms', 0) for r in results]
        stats[f'{ver}_avg_time_ms'] = np.mean([t for t in times if t > 0]) if any(t > 0 for t in times) else 0

    # By num_layers and num_devices breakdown
    stats['by_num_layers'] = {}
    for nl in sorted(set(r.num_layers for r in results)):
        sub = [r for r in results if r.num_layers == nl]
        stats['by_num_layers'][nl] = {
            'dp_tpot': [r.dp_tpot for r in sub],
            'beam_tpot': [r.beam_tpot for r in sub],
        }
        for ver in eval_versions:
            tpots = [getattr(r, f'{ver}_tpot') for r in sub]
            if tpots:
                stats['by_num_layers'][nl][f'{ver}_tpot'] = tpots

    stats['by_num_devices'] = {}
    for nd in sorted(set(r.num_devices for r in results)):
        sub = [r for r in results if r.num_devices == nd]
        stats['by_num_devices'][nd] = {
            'dp_tpot': [r.dp_tpot for r in sub],
            'beam_tpot': [r.beam_tpot for r in sub],
        }
        for ver in eval_versions:
            tpots = [getattr(r, f'{ver}_tpot') for r in sub]
            if tpots:
                stats['by_num_devices'][nd][f'{ver}_tpot'] = tpots

    stats['total_tests'] = total
    return stats
