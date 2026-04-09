"""All visualization functions for evaluation results."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Optional

from environment import compute_simple_tpot, create_random_config, compute_pipeline_tpot
from evaluation.runner import TestResult
from training.base_trainer import TrainingMetrics


# Color scheme for methods
METHOD_COLORS = {
    'v1': '#e74c3c',    # red
    'v2': '#3498db',    # blue
    'v3': '#2ecc71',    # green
    'v4': '#f39c12',    # orange
    'v5': '#9b59b6',    # purple
    'v6': '#e91e63',    # pink
    'v7': '#ff5722',    # deep orange
    'beam': '#1abc9c',  # teal
    'dp': '#34495e',    # dark gray
    'dp_raw': '#7f8c8d',  # medium gray
    'greedy': '#bdc3c7', # light gray
}
METHOD_NAMES = {
    'v1': 'V1-DQN',
    'v2': 'V2-PPO-Binary',
    'v3': 'V3-PPO-Order',
    'v4': 'V4-PPO-AutoReg',
    'v5': 'V5-MaskPPO',
    'v6': 'V6-GNN-PPO',
    'v7': 'V7-GNN-AR-PPO',
    'beam': 'BeamSearch',
    'dp': 'DP-Sorted',
    'dp_raw': 'DP-Raw',
    'greedy': 'Greedy',
}


def generate_all_plots(
    results: list,
    stats: dict,
    metrics_dict: dict,
    output_dir: str,
    config=None,
):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    print("  Generating training curves...")
    plot_training_curves(metrics_dict, output_dir)

    print("  Generating TPOT comparison...")
    plot_tpot_comparison(results, output_dir)

    print("  Generating device allocation heatmaps...")
    plot_device_allocation(results, output_dir)

    print("  Generating pipeline bubble visualization...")
    plot_pipeline_bubble(results, output_dir)

    print("  Generating scaling analysis...")
    plot_scaling_analysis(results, output_dir)

    print("  Generating strategy time comparison...")
    plot_strategy_time(results, output_dir)

    print("  Generating optimality verification...")
    plot_optimality_verification(results, output_dir)

    print("  Generating summary stats...")
    plot_summary_stats(results, stats, output_dir)

    print(f"  All plots saved to {output_dir}/")


# ============= 1. Training Curves =============

def plot_training_curves(metrics_dict: dict, output_dir: str):
    """Plot training curves for all versions (2x3 grid)."""
    versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: reward, TPOT, eval gap to DP
    # Row 2: policy loss, value loss, entropy

    for ver in versions:
        if ver not in metrics_dict:
            continue
        m = metrics_dict[ver]
        color = METHOD_COLORS.get(ver, 'black')

        if m.episode_rewards:
            # Smooth for readability
            smoothed = _smooth(m.episode_rewards, 50)
            axes[0, 0].plot(smoothed, color=color, label=METHOD_NAMES[ver], alpha=0.8)
        if m.episode_tpot:
            smoothed = _smooth(m.episode_tpot, 50)
            axes[0, 1].plot(smoothed, color=color, alpha=0.8)
        if m.eval_tpot and m.dp_tpot:
            gap = [(e - d) / d * 100 for e, d in zip(m.eval_tpot, m.dp_tpot)]
            axes[0, 2].plot(gap, color=color, alpha=0.8, marker='o', markersize=3)
        if m.policy_losses:
            smoothed = _smooth(m.policy_losses, 20)
            axes[1, 0].plot(smoothed, color=color, alpha=0.8)
        if m.value_losses:
            smoothed = _smooth(m.value_losses, 20)
            axes[1, 1].plot(smoothed, color=color, alpha=0.8)
        if m.entropies:
            smoothed = _smooth(m.entropies, 20)
            axes[1, 2].plot(smoothed, color=color, alpha=0.8)

    titles = ['Episode Reward', 'Episode TPOT', 'Eval Gap to DP (%)',
              'Policy Loss', 'Value Loss', 'Entropy']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Update')
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()


# ============= 2. Pipeline Bubble Visualization =============

def plot_pipeline_bubble(results: list, output_dir: str):
    """Gantt-chart style pipeline bubble visualization for one representative case."""

    # Pick a representative test case (mid-range)
    mid_idx = len(results) // 2
    r = results[mid_idx]
    devices, layers, ts = create_random_config(r.num_layers, r.num_devices, seed=r.seed)

    methods = {
        'V1-DQN': r.v1_partition,
        'V2-PPO': r.v2_partition,
        'V3-PPO': r.v3_partition,
        'V4-PPO': r.v4_partition,
        'V5-MaskPPO': r.v5_partition,
        'BeamSearch': r.beam_partition,
        'DP-Sorted': r.dp_partition,
        'DP-Raw': r.dp_raw_partition,
    }
    fig, axes = plt.subplots(len(methods), 1, figsize=(16, 3 * len(methods)))
    if len(methods) == 1:
        axes = [axes]

    stage_colors = plt.cm.Set3(np.linspace(0, 1, 12))

    num_microbatches = 4

    for ax, (name, partition) in zip(axes, methods.items()):
        if not partition:
            ax.set_title(f'{name} (no partition)')
            continue

        pipeline = compute_pipeline_tpot(partition, devices, layers, ts, num_microbatches)
        stage_times = pipeline.get('stage_times', [])
        num_stages = pipeline.get('num_stages', 1)

        # Extract stage device indices from partition
        stage_devs = []
        cur_dev = partition[0]
        for d in partition[1:]:
            if d != cur_dev:
                stage_devs.append(cur_dev)
                cur_dev = d
        stage_devs.append(cur_dev)

        # Draw Gantt chart
        for mb in range(num_microbatches):
            t = 0.0
            for s_idx in range(min(num_stages, len(stage_times))):
                start = t
                duration = stage_times[s_idx]
                color = stage_colors[s_idx % len(stage_colors)]
                ax.barh(mb, duration, left=start, height=0.6, color=color,
                        edgecolor='black', linewidth=0.5)

                # Transfer
                if s_idx < len(stage_devs) - 1:
                    from_dev = stage_devs[s_idx]
                    to_dev = stage_devs[s_idx + 1]
                    transfer_t = devices.transfer_time(from_dev, to_dev, ts)
                    ax.barh(mb, transfer_t, left=start + duration, height=0.6,
                            color='gray', alpha=0.4, hatch='///')
                    t = start + duration + transfer_t
                else:
                    t = start + duration

        tpot_val = compute_simple_tpot(partition, devices, layers, ts)

        ax.set_title(f'{name} (TPOT={tpot_val:.3f}, stages={num_stages})', fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Micro-batch')
        ax.set_yticks(range(num_microbatches))
        ax.set_yticklabels([f'MB{i}' for i in range(num_microbatches)])

    plt.suptitle(f'Pipeline Parallel Bubble (L={r.num_layers}, D={r.num_devices})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pipeline_bubble.png'), dpi=150)
    plt.close()


# ============= 3. Device Allocation Heatmap =============

def plot_device_allocation(results: list, output_dir: str):
    """Heatmap showing layer-to-device assignments."""
    mid_idx = len(results) // 2
    r = results[mid_idx]

    methods = {
        'V1-DQN': r.v1_partition,
        'V2-PPO': r.v2_partition,
        'V3-PPO': r.v3_partition,
        'V4-PPO': r.v4_partition,
        'V5-MaskPPO': r.v5_partition,
        'Beam': r.beam_partition,
        'DP-Sorted': r.dp_partition,
        'DP-Raw': r.dp_raw_partition,
    }

    # Filter to methods with partitions
    methods = {k: v for k, v in methods.items() if v}

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    cmap = plt.cm.tab20

    for ax, (name, partition) in zip(axes, methods.items()):
        nl = len(partition)
        nd = r.num_devices
        matrix = np.zeros((nd, nl))
        for l_idx, d_idx in enumerate(partition):
            if d_idx < nd:
                matrix[d_idx, l_idx] = 1

        ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest')
        ax.set_title(f'{name}', fontsize=10)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Device')
        ax.set_yticks(range(nd))
        ax.set_yticklabels([f'D{i}' for i in range(nd)])

    plt.suptitle(f'Device Allocation (L={r.num_layers}, D={r.num_devices})', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'device_allocation.png'), dpi=150)
    plt.close()


# ============= 4. TPOT Comparison Bar Chart =============

def plot_tpot_comparison(results: list, output_dir: str):
    """Grouped bar chart comparing TPOT across methods."""
    methods = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'beam', 'dp', 'dp_raw', 'greedy']
    method_labels = [METHOD_NAMES[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    # Average TPOT by num_layers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: avg TPOT per method
    avg_tpots = []
    for m in methods:
        tpots = [getattr(r, f'{m}_tpot') for r in results
                 if getattr(r, f'{m}_tpot', float('inf')) < float('inf')]
        avg_tpots.append(np.mean(tpots) if tpots else 0)

    bars = ax1.bar(range(len(methods)), avg_tpots, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Average TPOT')
    ax1.set_title('Average TPOT by Method')
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: gap to Beam Search
    beam_tpots = [r.beam_tpot for r in results]
    gaps = []
    for m in methods:
        tpots = [getattr(r, f'{m}_tpot') for r in results]
        gap = [(t - b) / b * 100 for t, b in zip(tpots, beam_tpots) if b > 0 and t < float('inf')]
        gaps.append(np.mean(gap) if gap else 0)

    ax2.bar(range(len(methods)), gaps, color=colors, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Gap to Beam Search (%)')
    ax2.set_title('Average Gap to Beam Search')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tpot_comparison.png'), dpi=150)
    plt.close()


# ============= 5. Scaling Analysis =============

def plot_scaling_analysis(results: list, output_dir: str):
    """Line plots: TPOT vs layers for different device counts."""
    methods = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'beam', 'dp', 'dp_raw']
    layer_values = sorted(set(r.num_layers for r in results))
    device_values = sorted(set(r.num_devices for r in results))

    # Pick representative device counts
    rep_devices = device_values[:5] if len(device_values) > 5 else device_values

    fig, axes = plt.subplots(2, len(rep_devices), figsize=(5 * len(rep_devices), 10))
    if len(rep_devices) == 1:
        axes = axes.reshape(2, 1)

    for col, nd in enumerate(rep_devices):
        # Top row: TPOT vs layers
        for m in methods:
            tpots_by_layer = {}
            for r in results:
                if r.num_devices == nd:
                    t = getattr(r, f'{m}_tpot', float('inf'))
                    if t < float('inf'):
                        tpots_by_layer.setdefault(r.num_layers, []).append(t)

            layers = sorted(tpots_by_layer.keys())
            avg_tpots = [np.mean(tpots_by_layer[l]) for l in layers]

            axes[0, col].plot(layers, avg_tpots, color=METHOD_COLORS[m],
                              marker='o', markersize=4, label=METHOD_NAMES[m], alpha=0.8)

        axes[0, col].set_title(f'{nd} Devices')
        axes[0, col].set_xlabel('Layers')
        axes[0, col].set_ylabel('TPOT')
        axes[0, col].grid(True, alpha=0.3)

        # Bottom row: inference time vs layers
        for m in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'beam']:
            times_by_layer = {}
            for r in results:
                if r.num_devices == nd:
                    t = getattr(r, f'{m}_time_ms', 0)
                    if t > 0:
                        times_by_layer.setdefault(r.num_layers, []).append(t)

            layers = sorted(times_by_layer.keys())
            avg_times = [np.mean(times_by_layer[l]) for l in layers]

            axes[1, col].plot(layers, avg_times, color=METHOD_COLORS[m],
                              marker='s', markersize=4, label=METHOD_NAMES[m], alpha=0.8)

        axes[1, col].set_title(f'{nd} Devices')
        axes[1, col].set_xlabel('Layers')
        axes[1, col].set_ylabel('Time (ms)')
        axes[1, col].grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=7, loc='upper left')
    axes[1, 0].legend(fontsize=7, loc='upper left')

    plt.suptitle('Scaling Analysis: TPOT and Inference Time vs Layers', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_analysis.png'), dpi=150)
    plt.close()


# ============= 6. Strategy Generation Time =============

def plot_strategy_time(results: list, output_dir: str):
    """Bar chart comparing inference time by method and problem size."""
    methods = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'beam', 'dp', 'dp_raw']
    layer_groups = sorted(set(r.num_layers for r in results))

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(layer_groups))
    width = 0.8 / len(methods)

    for i, m in enumerate(methods):
        times = []
        for nl in layer_groups:
            ts = [getattr(r, f'{m}_time_ms', 0) for r in results if r.num_layers == nl]
            times.append(np.mean(ts) if ts else 0)

        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, times, width, label=METHOD_NAMES[m],
               color=METHOD_COLORS[m], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Strategy Generation Time by Method and Problem Size')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layer_groups])
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_time.png'), dpi=150)
    plt.close()


# ============= 7. Optimality Verification =============

def plot_optimality_verification(results: list, output_dir: str):
    """Scatter plot + histogram showing how close each RL method gets to optimal."""
    rl_methods = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']

    # Filter to methods that have valid results
    valid_methods = []
    for m in rl_methods:
        m_tpots = [getattr(r, f'{m}_tpot', float('inf')) for r in results]
        if any(t < float('inf') for t in m_tpots):
            valid_methods.append(m)

    n_methods = len(valid_methods)
    if n_methods == 0:
        return

    # Layout: top row scatter plots, bottom-right histogram
    n_cols = min(n_methods + 1, 4)  # +1 for histogram
    n_rows = max(1, (n_methods + 1 + n_cols - 1) // n_cols)
    fig, axes_flat = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes_flat = np.array([axes_flat])
    axes_flat = axes_flat.flatten()

    # Scatter plots (DP TPOT vs method TPOT)
    for idx, m in enumerate(valid_methods):
        ax = axes_flat[idx]

        dp_tpots = [r.dp_tpot for r in results]
        m_tpots = [getattr(r, f'{m}_tpot', float('inf')) for r in results]

        # Filter valid
        valid = [(d, t) for d, t in zip(dp_tpots, m_tpots) if t < float('inf')]
        if valid:
            d_vals, t_vals = zip(*valid)
            ax.scatter(d_vals, t_vals, color=METHOD_COLORS[m], alpha=0.5, s=30)
            max_val = max(max(d_vals), max(t_vals))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='DP baseline')
            ax.set_xlabel('DP TPOT')
            ax.set_ylabel(f'{METHOD_NAMES[m]} TPOT')
            ax.set_title(f'{METHOD_NAMES[m]}')
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Use last subplot for histogram of gaps to Beam Search
    ax_hist = axes_flat[n_methods] if n_methods < len(axes_flat) else axes_flat[-1]
    for m in valid_methods:
        beam_tpots = [r.beam_tpot for r in results]
        m_tpots = [getattr(r, f'{m}_tpot', float('inf')) for r in results]
        gaps = [(t - b) / b * 100 for t, b in zip(m_tpots, beam_tpots)
                if b > 0 and t < float('inf')]
        if gaps:
            ax_hist.hist(gaps, bins=20, alpha=0.4, color=METHOD_COLORS[m],
                         label=METHOD_NAMES[m])

    ax_hist.axvline(x=0, color='black', linestyle='--')
    ax_hist.set_xlabel('Gap to Beam Search (%)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Gap Distribution to Beam Search')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_methods + 1, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('Optimality Verification', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimality_verification.png'), dpi=150)
    plt.close()


# ============= Summary Stats =============

def plot_summary_stats(results: list, stats: dict, output_dir: str):
    """Win rate bar chart and summary statistics."""
    rl_methods = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Win rate vs DP
    dp_wins = [stats.get(f'{m}_win_rate_dp', 0) for m in rl_methods]
    axes[0].bar(range(len(rl_methods)), dp_wins,
                color=[METHOD_COLORS[m] for m in rl_methods], edgecolor='black')
    axes[0].set_xticks(range(len(rl_methods)))
    axes[0].set_xticklabels([METHOD_NAMES[m] for m in rl_methods], rotation=45, ha='right')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_title('Win Rate vs DP (within 2%)')
    axes[0].axhline(y=80, color='red', linestyle='--', label='80% target')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Win rate vs Beam Search
    beam_wins = [stats.get(f'{m}_win_rate_beam', 0) for m in rl_methods]
    axes[1].bar(range(len(rl_methods)), beam_wins,
                color=[METHOD_COLORS[m] for m in rl_methods], edgecolor='black')
    axes[1].set_xticks(range(len(rl_methods)))
    axes[1].set_xticklabels([METHOD_NAMES[m] for m in rl_methods], rotation=45, ha='right')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_title('Win Rate vs Beam Search (within 2%)')
    axes[1].axhline(y=50, color='red', linestyle='--', label='50% target')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Average gap to Beam
    beam_gaps = [stats.get(f'{m}_avg_gap_to_beam', 0) for m in rl_methods]
    axes[2].bar(range(len(rl_methods)), beam_gaps,
                color=[METHOD_COLORS[m] for m in rl_methods], edgecolor='black')
    axes[2].set_xticks(range(len(rl_methods)))
    axes[2].set_xticklabels([METHOD_NAMES[m] for m in rl_methods], rotation=45, ha='right')
    axes[2].set_ylabel('Avg Gap (%)')
    axes[2].set_title('Average Gap to Beam Search')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Summary Statistics', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_stats.png'), dpi=150)
    plt.close()


# ============= Utility =============

def _smooth(data: list, window: int = 50) -> list:
    """Moving average smoothing."""
    if len(data) < window:
        return data
    arr = np.array(data)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode='valid')
    return smoothed.tolist()
