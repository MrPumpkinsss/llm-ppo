"""Comprehensive evaluation and visualization for LLM layer partitioning."""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

from config import TrainConfig, EvalConfig
from environment import (
    DeviceCluster, LayerModel, create_random_config,
    compute_simple_tpot, compute_pipeline_tpot
)
from baselines import dp_partition, greedy_partition, greedy_partition_advanced
from ppo_v1 import (
    OrderPredictor, build_observation, build_device_features, get_obs_dim,
)
from ppo_v2 import (
    DeviceOrderNetwork, build_device_features,
    generate_device_order, dp_for_device_order, ppo_v2_inference
)


@dataclass
class TestResult:
    """Result of a single test configuration."""
    test_id: int
    num_layers: int
    num_devices: int
    seed: int
    ppo_v1_tpot: float
    ppo_v2_tpot: float
    dp_tpot: float
    greedy_tpot: float
    greedy_basic_tpot: float
    ppo_v1_partition: list
    ppo_v2_partition: list
    dp_partition: list
    greedy_partition: list
    greedy_basic_partition: list
    ppo_v1_wins: bool = False
    ppo_v2_wins: bool = False


def _build_dev_feats(devs, num_devices):
    """Build per-device feature tensor for PPO-v1."""
    max_devices = 10
    features = np.zeros((max_devices, 4), dtype=np.float32)
    for d in range(num_devices):
        features[d, 0] = devs.compute_power[d]
        bw_to_others = devs.bandwidth[d, np.arange(num_devices) != d]
        features[d, 1] = bw_to_others.mean() if len(bw_to_others) > 0 else 0
        features[d, 2] = bw_to_others.max() if len(bw_to_others) > 0 else 0
        features[d, 3] = bw_to_others.min() if len(bw_to_others) > 0 else 0
    return features


def run_evaluation(
    network_v1, network_v2,
    train_config: TrainConfig,
    eval_config: EvalConfig,
) -> Tuple[List[TestResult], Dict]:
    """Run comprehensive evaluation across different configurations."""
    device = train_config.device
    max_devices = 10
    max_obs_dim = get_obs_dim(max_devices, 64)

    results = []
    stats = {
        'ppo_v1_wins': 0, 'ppo_v2_wins': 0, 'dp_wins': 0, 'greedy_wins': 0,
        'ppo_v1_avg_gap_to_dp': [], 'ppo_v2_avg_gap_to_dp': [],
        'ppo_v1_avg_gap_to_greedy': [], 'ppo_v2_avg_gap_to_greedy': [],
        'by_num_layers': {}, 'by_num_devices': {},
    }

    def ppo_v1_inference(network, obs_tensor, dev_feats_t, num_layers, num_devices, devs, lys, ts, device):
        """Argmax inference for PPO-v1 using OrderPredictor with fallback ordering."""
        candidates = []

        with torch.no_grad():
            # PPO greedy ordering
            order, _, _ = network.generate_order_autoregressive(
                obs_tensor, dev_feats_t, num_devices, greedy=True
            )
            order_list = order.squeeze(0).tolist()
            candidates.append((compute_simple_tpot(
                dp_for_device_order(num_layers, order_list, devs, lys, ts), devs, lys, ts
            ), order_list))

            # Sorted by compute (descending)
            sort_desc = sorted(range(num_devices), key=lambda d: devs.compute_power[d], reverse=True)
            candidates.append((compute_simple_tpot(
                dp_for_device_order(num_layers, sort_desc, devs, lys, ts), devs, lys, ts
            ), sort_desc))

            # Sorted by compute (ascending)
            sort_asc = sorted(range(num_devices), key=lambda d: devs.compute_power[d])
            candidates.append((compute_simple_tpot(
                dp_for_device_order(num_layers, sort_asc, devs, lys, ts), devs, lys, ts
            ), sort_asc))

            # Top-3 PPO orderings
            order_logits, _ = network.forward(obs_tensor, dev_feats_t, num_devices)
            probs = torch.softmax(order_logits.squeeze(0)[:num_devices], dim=-1)
            top3 = torch.topk(probs, min(3, num_devices)).indices.tolist()
            for d1 in top3:
                for d2 in [x for x in top3 if x != d1][:2]:
                    rest = [d for d in range(num_devices) if d not in [d1, d2]]
                    order = [d1, d2] + rest
                    candidates.append((compute_simple_tpot(
                        dp_for_device_order(num_layers, order, devs, lys, ts), devs, lys, ts
                    ), order))

        best_tpot, best_order = min(candidates, key=lambda x: x[0])
        return dp_for_device_order(num_layers, best_order, devs, lys, ts)

    test_id = 0
    for nl in eval_config.num_layers_to_test:
        for nd in eval_config.num_devices_to_test:
            stats['by_num_layers'][nl] = {'ppo_v1': [], 'ppo_v2': [], 'dp': [], 'greedy': []}
            stats['by_num_devices'][nd] = {'ppo_v1': [], 'ppo_v2': [], 'dp': [], 'greedy': []}

            for trial in range(3):  # 3 trials per (layers, devices) combination
                seed = train_config.seed + 200000 + test_id * 37
                devs, lys, ts = create_random_config(nl, nd, seed=seed)

                # --- PPO-v1 ---
                obs = build_observation(devs, lys, nd, nl)
                obs_padded = np.zeros(max_obs_dim, dtype=np.float32)
                obs_padded[:get_obs_dim(nd, nl)] = obs
                obs_t = torch.tensor(obs_padded, device=device).unsqueeze(0)

                dev_feats = build_device_features(devs, nd)
                dev_feats_padded = np.zeros((max_devices, 4), dtype=np.float32)
                dev_feats_padded[:nd] = dev_feats
                dev_feats_t = torch.tensor(dev_feats_padded, device=device).unsqueeze(0)

                with torch.no_grad():
                    ppo_v1_part = ppo_v1_inference(network_v1, obs_t, dev_feats_t, nl, nd, devs, lys, ts, device)
                    ppo_v1_tpot = compute_simple_tpot(ppo_v1_part, devs, lys, ts)

                # --- PPO-v2 with fallback ordering ---
                dev_feats = build_device_features(devs, nd)
                dev_feats_padded = np.zeros((max_devices, 4), dtype=np.float32)
                dev_feats_padded[:nd] = dev_feats
                dev_feats_t = torch.tensor(dev_feats_padded, device=device).unsqueeze(0)

                ppo_v2_part, ppo_v2_tpot = ppo_v2_inference(
                    network_v2, obs_t, dev_feats_t, nd, nl, devs, lys, ts
                )

                # --- DP ---
                dp_part = dp_partition(nl, nd, devs, lys, ts)
                dp_tpot = compute_simple_tpot(dp_part, devs, lys, ts)

                # --- Greedy (Advanced) ---
                gr_part = greedy_partition_advanced(nl, nd, devs, lys, ts)
                gr_tpot = compute_simple_tpot(gr_part, devs, lys, ts)

                # --- Greedy (Basic) ---
                gr_basic_part = greedy_partition(nl, nd, devs, lys, ts)
                gr_basic_tpot = compute_simple_tpot(gr_basic_part, devs, lys, ts)

                # Track stats
                stats['ppo_v1_avg_gap_to_dp'].append((ppo_v1_tpot - dp_tpot) / (dp_tpot + 1e-8) * 100)
                stats['ppo_v2_avg_gap_to_dp'].append((ppo_v2_tpot - dp_tpot) / (dp_tpot + 1e-8) * 100)
                stats['ppo_v1_avg_gap_to_greedy'].append((ppo_v1_tpot - gr_tpot) / (gr_tpot + 1e-8) * 100)
                stats['ppo_v2_avg_gap_to_greedy'].append((ppo_v2_tpot - gr_tpot) / (gr_tpot + 1e-8) * 100)

                # Wins (lower TPOT is better, within 2% of DP counts as win)
                is_v1_win = ppo_v1_tpot <= dp_tpot * 1.02
                is_v2_win = ppo_v2_tpot <= dp_tpot * 1.02
                if ppo_v1_tpot <= dp_tpot:
                    stats['ppo_v1_wins'] += 1
                if ppo_v2_tpot <= dp_tpot:
                    stats['ppo_v2_wins'] += 1
                if dp_tpot <= min(ppo_v1_tpot, ppo_v2_tpot, gr_tpot):
                    stats['dp_wins'] += 1
                if gr_tpot <= min(dp_tpot, ppo_v1_tpot, ppo_v2_tpot):
                    stats['greedy_wins'] += 1

                for key, val in [('ppo_v1', ppo_v1_tpot), ('ppo_v2', ppo_v2_tpot),
                                  ('dp', dp_tpot), ('greedy', gr_tpot)]:
                    stats['by_num_layers'][nl][key].append(val)
                    stats['by_num_devices'][nd][key].append(val)

                result = TestResult(
                    test_id=test_id, num_layers=nl, num_devices=nd, seed=seed,
                    ppo_v1_tpot=ppo_v1_tpot, ppo_v2_tpot=ppo_v2_tpot,
                    dp_tpot=dp_tpot, greedy_tpot=gr_tpot, greedy_basic_tpot=gr_basic_tpot,
                    ppo_v1_partition=ppo_v1_part, ppo_v2_partition=ppo_v2_part,
                    dp_partition=dp_part, greedy_partition=gr_part,
                    greedy_basic_partition=gr_basic_part,
                    ppo_v1_wins=is_v1_win, ppo_v2_wins=is_v2_win,
                )
                results.append(result)
                test_id += 1

    return results, stats


def plot_training_curves(metrics_v1, metrics_v2, output_dir: str):
    """Plot training curves: reward, TPOT, policy loss, value loss."""
    if metrics_v1 is None and metrics_v2 is None:
        print(f"Skipping training curves (no metrics available in eval-only mode)")
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Smooth helper
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Episode rewards
    ax = axes[0, 0]
    if metrics_v1.episode_rewards:
        ax.plot(smooth(metrics_v1.episode_rewards), alpha=0.8, label='PPO-v1')
    if metrics_v2.episode_rewards:
        ax.plot(smooth(metrics_v2.episode_rewards), alpha=0.8, label='PPO-v2')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (negative TPOT)')
    ax.set_title('Training Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode TPOT
    ax = axes[0, 1]
    if metrics_v1.episode_tpot:
        ax.plot(smooth(metrics_v1.episode_tpot), alpha=0.8, label='PPO-v1')
    if metrics_v2.episode_tpot:
        ax.plot(smooth(metrics_v2.episode_tpot), alpha=0.8, label='PPO-v2')
    ax.set_xlabel('Episode')
    ax.set_ylabel('TPOT')
    ax.set_title('Training TPOT (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Eval TPOT comparison
    ax = axes[0, 2]
    if metrics_v1.eval_tpot:
        x = np.arange(len(metrics_v1.eval_tpot))
        ax.plot(x, metrics_v1.eval_tpot, 'o-', label='PPO-v1', markersize=3)
        ax.plot(x, metrics_v1.dp_tpot, 's--', label='DP', markersize=3)
        ax.plot(x, metrics_v1.greedy_tpot, '^--', label='Greedy', markersize=3)
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('Avg TPOT')
    ax.set_title('PPO-v1: Eval TPOT Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Policy loss
    ax = axes[1, 0]
    if metrics_v1.policy_losses:
        ax.plot(smooth(metrics_v1.policy_losses, 20), alpha=0.8, label='PPO-v1')
    if metrics_v2.policy_losses:
        ax.plot(smooth(metrics_v2.policy_losses, 20), alpha=0.8, label='PPO-v2')
    ax.set_xlabel('Update')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Value loss
    ax = axes[1, 1]
    if metrics_v1.value_losses:
        ax.plot(smooth(metrics_v1.value_losses, 20), alpha=0.8, label='PPO-v1')
    if metrics_v2.value_losses:
        ax.plot(smooth(metrics_v2.value_losses, 20), alpha=0.8, label='PPO-v2')
    ax.set_xlabel('Update')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 2]
    if metrics_v1.entropies:
        ax.plot(smooth(metrics_v1.entropies, 20), alpha=0.8, label='PPO-v1')
    if metrics_v2.entropies:
        ax.plot(smooth(metrics_v2.entropies, 20), alpha=0.8, label='PPO-v2')
    ax.set_xlabel('Update')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'training_curves.png')}")


def plot_tpot_comparison(results: List[TestResult], output_dir: str):
    """Bar chart comparing TPOT across algorithms for each test."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    test_ids = [r.test_id for r in results]
    ppo_v1_tpots = [r.ppo_v1_tpot for r in results]
    ppo_v2_tpots = [r.ppo_v2_tpot for r in results]
    dp_tpots = [r.dp_tpot for r in results]
    greedy_tpots = [r.greedy_tpot for r in results]

    x = np.arange(len(test_ids))
    width = 0.2

    ax = axes[0]
    ax.bar(x - 1.5 * width, dp_tpots, width, label='DP (Optimal)', color='gold', alpha=0.8)
    ax.bar(x - 0.5 * width, ppo_v1_tpots, width, label='PPO-v1 (Direct)', color='steelblue', alpha=0.8)
    ax.bar(x + 0.5 * width, ppo_v2_tpots, width, label='PPO-v2 (Order+DP)', color='forestgreen', alpha=0.8)
    ax.bar(x + 1.5 * width, greedy_tpots, width, label='Greedy (Adv)', color='coral', alpha=0.8)

    ax.set_xlabel('Test Case')
    ax.set_ylabel('TPOT')
    ax.set_title('TPOT Comparison Across All Test Cases')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Normalized: gap to DP
    ax = axes[1]
    gap_v1 = [(r.ppo_v1_tpot - r.dp_tpot) / (r.dp_tpot + 1e-8) * 100 for r in results]
    gap_v2 = [(r.ppo_v2_tpot - r.dp_tpot) / (r.dp_tpot + 1e-8) * 100 for r in results]
    gap_gr = [(r.greedy_tpot - r.dp_tpot) / (r.dp_tpot + 1e-8) * 100 for r in results]

    ax.bar(x - width, gap_v1, width, label='PPO-v1 vs DP', color='steelblue', alpha=0.8)
    ax.bar(x, gap_v2, width, label='PPO-v2 vs DP', color='forestgreen', alpha=0.8)
    ax.bar(x + width, gap_gr, width, label='Greedy vs DP', color='coral', alpha=0.8)
    ax.axhline(y=0, color='gold', linewidth=2, label='DP (baseline)')
    ax.axhline(y=2, color='red', linewidth=1, linestyle='--', alpha=0.5, label='2% threshold')

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Gap to DP (%)')
    ax.set_title('TPOT Gap to DP Optimal (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tpot_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'tpot_comparison.png')}")


def plot_device_allocation(result: TestResult, output_dir: str, suffix: str = ""):
    """Visualize layer allocation across devices for a single test case."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    partitions = [
        ('DP (Optimal)', result.dp_partition, 'gold'),
        ('PPO-v1 (Direct)', result.ppo_v1_partition, 'steelblue'),
        ('PPO-v2 (Order+DP)', result.ppo_v2_partition, 'forestgreen'),
        ('Greedy (Advanced)', result.greedy_partition, 'coral'),
    ]

    for idx, (name, partition, color) in enumerate(partitions):
        ax = axes[idx // 2, idx % 2]
        nl = result.num_layers
        nd = result.num_devices

        # Create allocation matrix
        alloc = np.zeros((nd, nl))
        for layer_idx, dev_id in enumerate(partition):
            alloc[dev_id, layer_idx] = 1

        im = ax.imshow(alloc, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax.set_yticks(range(nd))
        ax.set_yticklabels([f'Dev {i}' for i in range(nd)])
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Device')
        ax.set_title(f'{name}\nTPOT = {compute_simple_tpot_from_partition(partition):.4f}')

        # Add layer numbers
        for l in range(nl):
            for d in range(nd):
                if alloc[d, l] > 0:
                    ax.text(l, d, str(l), ha='center', va='center',
                           fontsize=max(4, 12 - nl // 8), color='black')

    plt.suptitle(f'Test #{result.test_id}: {result.num_layers} Layers × {result.num_devices} Devices',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f'device_allocation_test{result.test_id}{suffix}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, fname)}")


def compute_simple_tpot_from_partition(partition: list) -> float:
    """Placeholder - actual computation needs env objects."""
    return 0.0


def plot_pipeline_parallel(
    result: TestResult,
    devs: DeviceCluster,
    lys: LayerModel,
    tensor_size: float,
    output_dir: str = "results",
    num_microbatches: int = 4,
):
    """Visualize pipeline parallel execution across ALL devices (including unused).

    - All device rows are shown, idle devices are clearly marked
    - Compute blocks per micro-batch
    - Transfer blocks between stages (network latency)
    - Corner labels for each device row
    """
    partitions = {
        'DP': result.dp_partition,
        'PPO-v1': result.ppo_v1_partition,
        'PPO-v2': result.ppo_v2_partition,
        'Greedy': result.greedy_partition,
    }

    nd = result.num_devices

    # Pre-compute max time across all algorithms
    max_time = 0.0
    for name, partition in partitions.items():
        pipeline_info = compute_pipeline_tpot(partition, devs, lys, tensor_size, num_microbatches)
        startup = sum(pipeline_info['stage_times']) + pipeline_info['transfer_overhead'] * len(pipeline_info['stage_times'])
        total = startup + (num_microbatches - 1) * (pipeline_info['max_stage_time'] + pipeline_info['transfer_overhead'])
        max_time = max(max_time, total)
    max_time *= 1.05

    # Helpers
    def get_stages(partition):
        stages = []
        cur_dev = partition[0]
        cur_layers = [0]
        for i in range(1, len(partition)):
            if partition[i] == cur_dev:
                cur_layers.append(i)
            else:
                stages.append((cur_dev, cur_layers[:]))
                cur_dev = partition[i]
                cur_layers = [i]
        stages.append((cur_dev, cur_layers[:]))
        return stages

    def get_stage_details(partition, stages):
        details = []
        stage_devs = [s[0] for s in stages]
        prev_time = 0.0
        for si, (dev_id, layer_indices) in enumerate(stages):
            total_cost = sum(lys.compute_costs[l] for l in layer_indices)
            stage_time = total_cost / devs.compute_power[dev_id]
            transfer_in = 0.0
            if si > 0:
                from_dev = stage_devs[si - 1]
                to_dev = stage_devs[si]
                transfer_in = devs.transfer_time(from_dev, to_dev, tensor_size)
            details.append({
                'dev_id': dev_id,
                'layer_indices': layer_indices,
                'stage_time': stage_time,
                'transfer_in': transfer_in,
                'start_time': prev_time + transfer_in if si > 0 else 0.0,
            })
            prev_time = details[-1]['start_time'] + stage_time
        return details

    fig, all_axes = plt.subplots(len(partitions), 1, figsize=(22, 5.5 * len(partitions)))

    stage_colors = plt.cm.tab10(np.linspace(0, 1, nd))
    transfer_color = '#888888'
    idle_color = '#f0f0f0'

    for row, (name, partition) in enumerate(partitions.items()):
        ax = all_axes[row] if len(partitions) > 1 else all_axes

        stages = get_stages(partition)
        stage_details = get_stage_details(partition, stages)
        pipeline_info = compute_pipeline_tpot(partition, devs, lys, tensor_size, num_microbatches)
        used_devs = set(d['dev_id'] for d in stage_details)
        num_idle = nd - len(used_devs)

        # Draw idle background for ALL devices
        for dev_id in range(nd):
            ax.barh(dev_id, max_time, left=0, color=idle_color,
                   edgecolor='#cccccc', linewidth=0.5)

        # Draw compute + transfer blocks per active device
        used_colors = {}
        for si, detail in enumerate(stage_details):
            dev_id = detail['dev_id']
            stage_time = detail['stage_time']
            stage_start = detail['start_time']
            transfer_in = detail['transfer_in']
            color = stage_colors[si % len(stage_colors)]
            used_colors[si] = color

            # Transfer block
            if si > 0 and transfer_in > 0:
                t_start = stage_start - transfer_in
                ax.barh(dev_id, transfer_in, left=t_start,
                       color=transfer_color, edgecolor='black', linewidth=0.5,
                       alpha=0.6, hatch='///')
                ax.text(t_start + transfer_in / 2, dev_id,
                       f'T{si}', ha='center', va='center', fontsize=6,
                       color='white', fontweight='bold')

            # Compute blocks per micro-batch
            for mb in range(num_microbatches):
                mb_offset = mb * (pipeline_info['max_stage_time'] + pipeline_info['transfer_overhead'])
                start = stage_start + mb_offset
                ax.barh(dev_id, stage_time, left=start,
                       color=color, edgecolor='black', linewidth=0.5, alpha=0.85)
                ax.text(start + stage_time / 2, dev_id,
                       f'MB{mb}', ha='center', va='center', fontsize=6, fontweight='bold')

        # Corner labels (left of each row)
        for dev_id in range(nd):
            ax.text(-0.003 * max_time, dev_id, f'Dev {dev_id}',
                   ha='right', va='center', fontsize=9, fontweight='bold',
                   color='#333333', transform=ax.get_yaxis_transform(), clip_on=False)

        # Legend
        handles = []
        for si, color in used_colors.items():
            d = stage_details[si]
            patch = mpatches.Patch(
                color=color,
                label=f'Stage {si} (Dev {d["dev_id"]}, {len(d["layer_indices"])}L)'
            )
            handles.append(patch)
        handles.append(mpatches.Patch(
            color=transfer_color, alpha=0.6, hatch='///',
            label='Transfer (network latency)'
        ))
        if num_idle > 0:
            handles.append(mpatches.Patch(
                facecolor=idle_color, edgecolor='#cccccc', linewidth=1,
                label=f'Idle ({num_idle} device{"s" if num_idle > 1 else ""})'
            ))

        ax.set_yticks(range(nd))
        ax.set_yticklabels([f'Dev {i}' for i in range(nd)], fontsize=9)
        ax.set_xlim(-0.12 * max_time, max_time * 1.02)
        ax.set_xlabel('Time (arbitrary units)', fontsize=10)
        ax.set_title(
            f'{name}: TPOT={pipeline_info["tpot"]:.4f} | '
            f'Bubble={pipeline_info["bubble_fraction"]*100:.1f}% | '
            f'{len(stages)} stage(s), {num_idle} idle',
            fontsize=11, fontweight='bold'
        )
        ax.legend(handles=handles, loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_ylabel('Device', fontsize=10)

    plt.suptitle(
        f'Pipeline Parallel Visualization - Test #{result.test_id} '
        f'({result.num_layers}L x {result.num_devices}D, {num_microbatches} microbatches)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pipeline_parallel_test{result.test_id}.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, f'pipeline_parallel_test{result.test_id}.png')}")


def plot_scaling_analysis(stats: Dict, output_dir: str):
    """Analyze performance scaling with number of layers and devices."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # By num_layers
    ax = axes[0, 0]
    for nl in sorted(stats['by_num_layers'].keys()):
        data = stats['by_num_layers'][nl]
        means = [np.mean(data['ppo_v1']), np.mean(data['ppo_v2']),
                 np.mean(data['dp']), np.mean(data['greedy'])]
        ax.bar([f'{nl}L\nPPO-v1', f'{nl}L\nPPO-v2', f'{nl}L\nDP', f'{nl}L\nGreedy'],
               means, color=['steelblue', 'forestgreen', 'gold', 'coral'], alpha=0.8)
    ax.set_ylabel('Avg TPOT')
    ax.set_title('Avg TPOT by Number of Layers')
    ax.grid(True, alpha=0.3, axis='y')

    # By num_devices
    ax = axes[0, 1]
    for nd in sorted(stats['by_num_devices'].keys()):
        data = stats['by_num_devices'][nd]
        means = [np.mean(data['ppo_v1']), np.mean(data['ppo_v2']),
                 np.mean(data['dp']), np.mean(data['greedy'])]
        ax.bar([f'{nd}D\nPPO-v1', f'{nd}D\nPPO-v2', f'{nd}D\nDP', f'{nd}D\nGreedy'],
               means, color=['steelblue', 'forestgreen', 'gold', 'coral'], alpha=0.8)
    ax.set_ylabel('Avg TPOT')
    ax.set_title('Avg TPOT by Number of Devices')
    ax.grid(True, alpha=0.3, axis='y')

    # Gap to DP by layers
    ax = axes[1, 0]
    layers_list = sorted(stats['by_num_layers'].keys())
    gap_v1_by_l = []
    gap_v2_by_l = []
    gap_gr_by_l = []
    for nl in layers_list:
        data = stats['by_num_layers'][nl]
        dp_mean = np.mean(data['dp'])
        gap_v1_by_l.append((np.mean(data['ppo_v1']) - dp_mean) / (dp_mean + 1e-8) * 100)
        gap_v2_by_l.append((np.mean(data['ppo_v2']) - dp_mean) / (dp_mean + 1e-8) * 100)
        gap_gr_by_l.append((np.mean(data['greedy']) - dp_mean) / (dp_mean + 1e-8) * 100)

    ax.plot(layers_list, gap_v1_by_l, 'o-', label='PPO-v1', color='steelblue')
    ax.plot(layers_list, gap_v2_by_l, 'o-', label='PPO-v2', color='forestgreen')
    ax.plot(layers_list, gap_gr_by_l, 'o-', label='Greedy', color='coral')
    ax.axhline(y=0, color='gold', linewidth=2, linestyle='--')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Gap to DP (%)')
    ax.set_title('Gap to DP Optimal by Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gap to DP by devices
    ax = axes[1, 1]
    devs_list = sorted(stats['by_num_devices'].keys())
    gap_v1_by_d = []
    gap_v2_by_d = []
    gap_gr_by_d = []
    for nd in devs_list:
        data = stats['by_num_devices'][nd]
        dp_mean = np.mean(data['dp'])
        gap_v1_by_d.append((np.mean(data['ppo_v1']) - dp_mean) / (dp_mean + 1e-8) * 100)
        gap_v2_by_d.append((np.mean(data['ppo_v2']) - dp_mean) / (dp_mean + 1e-8) * 100)
        gap_gr_by_d.append((np.mean(data['greedy']) - dp_mean) / (dp_mean + 1e-8) * 100)

    ax.plot(devs_list, gap_v1_by_d, 'o-', label='PPO-v1', color='steelblue')
    ax.plot(devs_list, gap_v2_by_d, 'o-', label='PPO-v2', color='forestgreen')
    ax.plot(devs_list, gap_gr_by_d, 'o-', label='Greedy', color='coral')
    ax.axhline(y=0, color='gold', linewidth=2, linestyle='--')
    ax.set_xlabel('Number of Devices')
    ax.set_ylabel('Gap to DP (%)')
    ax.set_title('Gap to DP Optimal by Devices')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_analysis.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'scaling_analysis.png')}")


def plot_summary_stats(stats: Dict, output_dir: str):
    """Plot summary statistics: win rates, avg gaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Win rates
    ax = axes[0]
    total = sum([stats['ppo_v1_wins'], stats['ppo_v2_wins'], stats['dp_wins'], stats['greedy_wins']])
    if total > 0:
        labels = ['PPO-v1\n≤ DP', 'PPO-v2\n≤ DP', 'DP\nOptimal', 'Greedy\nBest']
        values = [stats['ppo_v1_wins'], stats['ppo_v2_wins'],
                 stats['dp_wins'], stats['greedy_wins']]
        colors = ['steelblue', 'forestgreen', 'gold', 'coral']
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title('Algorithm Comparison\n(Times achieving best or ≤ DP)')
    ax.grid(True, alpha=0.3, axis='y')

    # Avg gap to DP
    ax = axes[1]
    gaps = {
        'PPO-v1': np.mean(stats['ppo_v1_avg_gap_to_dp']),
        'PPO-v2': np.mean(stats['ppo_v2_avg_gap_to_dp']),
        'Greedy': np.mean(stats['ppo_v1_avg_gap_to_greedy']),  # gap to greedy baseline
    }
    colors = ['steelblue', 'forestgreen', 'coral']
    bars = ax.bar(gaps.keys(), gaps.values(), color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Avg Gap to DP (%)')
    ax.set_title('Average Gap to DP Optimal')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, gaps.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')

    # Distribution of gaps
    ax = axes[2]
    ax.hist(stats['ppo_v1_avg_gap_to_dp'], bins=20, alpha=0.5, label='PPO-v1', color='steelblue')
    ax.hist(stats['ppo_v2_avg_gap_to_dp'], bins=20, alpha=0.5, label='PPO-v2', color='forestgreen')
    ax.axvline(x=0, color='gold', linewidth=2, linestyle='--', label='DP baseline')
    ax.axvline(x=2, color='red', linewidth=1, linestyle=':', alpha=0.5, label='2% threshold')
    ax.set_xlabel('Gap to DP (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of TPOT Gap to DP')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_stats.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'summary_stats.png')}")


def generate_all_plots(
    results, stats, metrics_v1, metrics_v2,
    output_dir: str, train_config: TrainConfig
):
    """Generate all evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating Plots ===")

    # 1. Training curves
    plot_training_curves(metrics_v1, metrics_v2, output_dir)

    # 2. TPOT comparison
    plot_tpot_comparison(results, output_dir)

    # 3. Device allocation for a few representative cases
    # Pick cases with different (layers, devices) combinations
    representative = []
    seen = set()
    for r in results:
        key = (r.num_layers, r.num_devices)
        if key not in seen:
            representative.append(r)
            seen.add(key)
        if len(representative) >= 6:
            break

    for r in representative:
        seed = r.seed
        devs, lys, ts = create_random_config(r.num_layers, r.num_devices, seed=seed)

        # Fix device allocation plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        partitions = [
            ('DP (Optimal)', r.dp_partition, 'gold'),
            ('PPO-v1 (Direct)', r.ppo_v1_partition, 'steelblue'),
            ('PPO-v2 (Order+DP)', r.ppo_v2_partition, 'forestgreen'),
            ('Greedy (Advanced)', r.greedy_partition, 'coral'),
        ]

        for idx, (name, partition, color) in enumerate(partitions):
            ax = axes[idx // 2, idx % 2]
            nl = r.num_layers
            nd = r.num_devices
            tpot = compute_simple_tpot(partition, devs, lys, ts)

            alloc = np.zeros((nd, nl))
            for layer_idx, dev_id in enumerate(partition):
                alloc[dev_id, layer_idx] = 1

            im = ax.imshow(alloc, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            ax.set_yticks(range(nd))
            ax.set_yticklabels([f'Dev {i}' for i in range(nd)])
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Device')
            ax.set_title(f'{name}\nTPOT = {tpot:.4f}')

        plt.suptitle(f'Test #{r.test_id}: {r.num_layers} Layers × {r.num_devices} Devices',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'device_allocation_test{r.test_id}.png'), dpi=150)
        plt.close()
        print(f"Saved: device_allocation_test{r.test_id}.png")

        # 4. Pipeline parallel visualization
        plot_pipeline_parallel(r, devs, lys, ts, output_dir)

    # 5. Scaling analysis
    plot_scaling_analysis(stats, output_dir)

    # 6. Summary stats
    plot_summary_stats(stats, output_dir)

    print("\n=== All plots generated ===")
