"""Scaling sweep: test V1-V5 across 24-64 layers and 4/6/8/10 devices."""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11

from config import TrainConfig, EvalConfig
from environment import create_random_config, compute_simple_tpot
from baselines import dp_partition, brute_force_optimal, beam_search_dp
from agents.dqn_v1 import DQNv1Network, dqn_v1_inference
from agents.ppo_v2 import PPOv2Network, ppo_v2_inference
from agents.ppo_v3 import PPOv3Network, ppo_v3_inference
from agents.ppo_v4 import PPOv4Network, ppo_v4_inference
from agents.maskable_ppo_v5 import MaskablePPOv5Network, maskable_v5_inference
from agents.gnn_ppo_v6 import PPOv6Network, ppo_v6_inference
from agents.gnn_ar_ppo_v7 import PPOv7Network, ppo_v7_inference


ALL_VERSIONS = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']


def load_all_models(model_dir='results', versions=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_device = torch.device(device)
    networks = {}

    version_classes = {
        'v1': DQNv1Network, 'v2': PPOv2Network,
        'v3': PPOv3Network, 'v4': PPOv4Network,
        'v5': MaskablePPOv5Network, 'v6': PPOv6Network,
        'v7': PPOv7Network,
    }
    if versions is None:
        versions = ALL_VERSIONS

    for ver in versions:
        NetClass = version_classes[ver]
        path = os.path.join(model_dir, f'{ver}_model.pt')
        if os.path.exists(path):
            net = NetClass().to(torch_device)
            net.load_state_dict(torch.load(path, map_location=torch_device))
            net.eval()
            networks[ver] = net
            print(f"  Loaded {ver} from {path}")
        else:
            print(f"  WARNING: {ver} not found at {path}")

    return networks, torch_device


def run_sweep(networks, torch_device, output_dir='results',
              layers_range=None, devices_range=None, runs=10, seeds=None, versions=None):
    if layers_range is None:
        layers_range = list(range(24, 65, 4))
    if devices_range is None:
        devices_range = [4, 6, 8, 10]
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024, 1337, 2048, 3141, 4096, 5555,
                 6789, 7777, 8888, 9999, 10101]
    if versions is None:
        versions = ALL_VERSIONS

    base_keys = ['layers', 'dp', 'brute', 't_dp', 't_brute']
    ver_keys = [f'{v}' for v in versions] + [f't_{v}' for v in versions]
    results = {nd: {k: [] for k in base_keys + ver_keys} for nd in devices_range}

    for nd in devices_range:
        do_brute = True  # always compute best-known: brute for ≤4, top-k for >4

        for nl in layers_range:
            tpot_dp_list, tpot_brute_list = [], []
            t_dp, t_brute = [], []
            tpot_ver = {v: [] for v in versions}
            t_ver = {v: [] for v in versions}

            # Inference functions per version
            infer_fns = {
                'v1': dqn_v1_inference, 'v2': ppo_v2_inference,
                'v3': ppo_v3_inference, 'v4': ppo_v4_inference,
                'v5': maskable_v5_inference, 'v6': ppo_v6_inference,
                'v7': ppo_v7_inference,
            }

            for r in range(runs):
                seed = seeds[r] if r < len(seeds) else seeds[-1] + r
                devs, lys, ts = create_random_config(nl, nd, seed=seed)

                # DP baseline
                t0 = time.time()
                dp_part = dp_partition(nl, nd, devs, lys, ts)
                dp_tpot = compute_simple_tpot(dp_part, devs, lys, ts)
                t_dp.append(time.time() - t0)
                tpot_dp_list.append(dp_tpot)

                # Best-known: brute-force for ≤4 devices, top-k beam for >4
                if do_brute:
                    t0 = time.time()
                    if nd <= 4:
                        _, brute_tpot = brute_force_optimal(nl, nd, devs, lys, ts)
                    else:
                        brute_part = beam_search_dp(nl, nd, devs, lys, ts, beam_width=2)
                        brute_tpot = compute_simple_tpot(brute_part, devs, lys, ts)
                    t_brute.append(time.time() - t0)
                    tpot_brute_list.append(brute_tpot)

                # RL versions
                for v in versions:
                    if v in networks:
                        t0 = time.time()
                        _, tpot = infer_fns[v](networks[v], devs, lys, ts, nl, nd, torch_device)
                        tpot_ver[v].append(tpot)
                        t_ver[v].append(time.time() - t0)

            rec = results[nd]
            rec['layers'].append(nl)
            for v in versions:
                rec[v].append(np.mean(tpot_ver[v]) if tpot_ver[v] else 0)
                rec[f't_{v}'].append(np.mean(t_ver[v]) * 1000 if t_ver[v] else 0)
            rec['dp'].append(np.mean(tpot_dp_list))
            rec['t_dp'].append(np.mean(t_dp) * 1000)

            if do_brute:
                rec['brute'].append(np.mean(tpot_brute_list))
                rec['t_brute'].append(np.mean(t_brute) * 1000)
            else:
                rec['brute'].append(None)
                rec['t_brute'].append(None)

            ver_str = ' '.join(f'{v}={np.mean(tpot_ver[v]):.3f}' for v in versions if tpot_ver[v])
            brute_str = f" Brute={np.mean(tpot_brute_list):.3f}" if do_brute else ""
            print(f"  {nd}D/{nl}L | {ver_str} DP={np.mean(tpot_dp_list):.3f}{brute_str}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'scaling_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results, layers_range, devices_range


def plot_sweep(results, layers_range, devices_range, output_dir, versions=None):
    if versions is None:
        versions = ALL_VERSIONS

    fig, axes = plt.subplots(2, len(devices_range), figsize=(5 * len(devices_range), 10))
    if len(devices_range) == 1:
        axes = axes.reshape(2, 1)

    colors = {'v1': '#3b82f6', 'v2': '#22c55e', 'v3': '#f97316', 'v4': '#a855f7',
              'v5': '#06b6d4', 'v6': '#e91e63', 'v7': '#8b5cf6', 'dp': '#f59e0b', 'brute': '#dc2626'}
    markers = {'v1': 'o', 'v2': 's', 'v3': 'D', 'v4': 'P', 'v5': 'X', 'v6': 'v', 'v7': 'H', 'dp': '^', 'brute': '*'}
    ver_labels = {'v1': 'V1-DQN', 'v2': 'V2-PPO', 'v3': 'V3-PPO',
                  'v4': 'V4-PPO', 'v5': 'V5-MaskPPO', 'v6': 'V6-GNN-PPO', 'v7': 'V7-AR-GNN'}

    # Row 0: Gap to brute-force or DP (%)
    for col, nd in enumerate(devices_range):
        ax = axes[0, col]
        r = results[nd]
        ls = r['layers']
        dp = np.array(r['dp'])

        # Use brute-force as baseline if available, otherwise DP
        baseline_key = 'brute' if r['brute'][0] is not None else 'dp'
        baseline = np.array([v if v is not None else dp[i] for i, v in enumerate(r['brute'])])

        for v in versions:
            gap = (np.array(r[v]) - baseline) / baseline * 100
            ax.plot(ls, gap, marker=markers[v], color=colors[v],
                    label=ver_labels[v], linewidth=2, markersize=5)

        baseline_label = 'Brute-force optimal' if baseline_key == 'brute' else 'DP baseline'
        ax.axhline(y=0, color=colors[baseline_key], linewidth=2, linestyle='--', label=baseline_label)
        # Draw DP gap line if brute-force is the baseline
        if baseline_key == 'brute':
            dp_gap = (dp - baseline) / baseline * 100
            ax.plot(ls, dp_gap, marker=markers['dp'], color=colors['dp'],
                    label='DP', linewidth=2, markersize=5, linestyle='--')
        ax.set_xlabel('Layers')
        ax.set_ylabel(f'Gap to {baseline_key.title()} (%)')
        ax.set_title(f'{nd} Devices — Gap to {baseline_key.title()}')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Row 1: Raw TPOT
    for col, nd in enumerate(devices_range):
        ax = axes[1, col]
        r = results[nd]
        ls = r['layers']

        # Brute-force curve if available
        brute_vals = [v for v in r['brute'] if v is not None]
        if brute_vals:
            ax.plot(ls, brute_vals, marker=markers['brute'], color=colors['brute'],
                    label='Brute-force', linewidth=2.5, markersize=8, linestyle='-')

        ax.plot(ls, r['dp'], marker=markers['dp'], color=colors['dp'],
                label='DP', linewidth=2, markersize=6)
        for v in versions:
            ax.plot(ls, r[v], marker=markers[v], color=colors[v],
                    label=ver_labels[v], linewidth=2, markersize=5)

        ax.set_xlabel('Layers')
        ax.set_ylabel('Avg TPOT')
        ax.set_title(f'{nd} Devices — Raw TPOT')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ver_str = '+'.join(versions)
    plt.suptitle(f'{ver_str} vs DP Scaling\n(avg of 10 runs)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, 'scaling_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


def plot_timing(results, layers_range, devices_range, output_dir, versions=None):
    """Separate timing plot: inference time vs layers for each algorithm."""
    if versions is None:
        versions = ALL_VERSIONS

    fig, axes = plt.subplots(1, len(devices_range), figsize=(5 * len(devices_range), 5))
    if len(devices_range) == 1:
        axes = [axes]

    colors = {'v1': '#3b82f6', 'v2': '#22c55e', 'v3': '#f97316', 'v4': '#a855f7',
              'v5': '#06b6d4', 'v6': '#e91e63', 'v7': '#8b5cf6', 'dp': '#f59e0b', 'brute': '#dc2626'}
    markers = {'v1': 'o', 'v2': 's', 'v3': 'D', 'v4': 'P', 'v5': 'X', 'v6': 'v', 'v7': 'H', 'dp': '^', 'brute': '*'}
    ver_labels = {'v1': 'V1-DQN', 'v2': 'V2-PPO', 'v3': 'V3-PPO',
                  'v4': 'V4-PPO', 'v5': 'V5-MaskPPO', 'v6': 'V6-GNN-PPO', 'v7': 'V7-AR-GNN'}

    for col, nd in enumerate(devices_range):
        ax = axes[col]
        r = results[nd]
        ls = r['layers']

        # Brute-force timing if available
        brute_times = [v for v in r['t_brute'] if v is not None]
        if brute_times:
            ax.plot(ls, brute_times, marker=markers['brute'], color=colors['brute'],
                    label='Brute-force', linewidth=2, markersize=6)

        ax.plot(ls, r['t_dp'], marker=markers['dp'], color=colors['dp'],
                label='DP', linewidth=2, markersize=6)
        for v in versions:
            ax.plot(ls, r[f't_{v}'], marker=markers[v], color=colors[v],
                    label=ver_labels[v], linewidth=2, markersize=5)

        ax.set_xlabel('Layers')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{nd} Devices — Inference Time')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('Inference Time vs Layers (avg of 10 runs)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, 'scaling_timing.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def print_gap_table(results, layers_range, devices_range, versions=None):
    if versions is None:
        versions = ALL_VERSIONS

    print("\n" + "=" * 170)
    print("GAP TO DP (%)" + ("  — Brute-force gap shown as DP gap" if any(results[nd]['brute'][0] is not None for nd in devices_range) else ""))
    print(f"{'Layers':>8}", end='')
    for nd in devices_range:
        has_brute = results[nd]['brute'][0] is not None
        brute_col = f" {nd}D-DP " if has_brute else ""
        ver_cols = ' '.join(f'{nd}D-{v}' for v in versions)
        print(f"  {ver_cols}{brute_col}", end='')
    print()
    for li, nl in enumerate(layers_range):
        print(f"{nl:>8}", end='')
        for nd in devices_range:
            r = results[nd]
            dp = r['dp'][li]
            for v in versions:
                vg = (r[v][li] - dp) / dp * 100
                print(f" {vg:+6.2f}", end='')
            if r['brute'][li] is not None:
                dpg = (dp - r['brute'][li]) / r['brute'][li] * 100
                print(f" {dpg:+6.2f}", end='')
        print()


def main():
    parser = argparse.ArgumentParser(description='Scaling Sweep')
    parser.add_argument('--model-dir', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--seeds', type=int, default=15)
    args = parser.parse_args()

    eval_config = EvalConfig()
    versions = eval_config.eval_versions
    print(f"Evaluating versions: {versions}")

    print("Loading models...")
    networks, torch_device = load_all_models(args.model_dir, versions=versions)

    if not networks:
        print("No models found. Train first with python main.py")
        sys.exit(1)

    layers_range = list(range(24, 65, 4))
    devices_range = [2, 4, 6, 10]
    seeds = [42, 123, 456, 789, 1024, 1337, 2048, 3141, 4096, 5555,
             6789, 7777, 8888, 9999, 10101][:args.seeds]

    print(f"\nSweep: layers={layers_range}, devices={devices_range}, runs={args.seeds}")
    print("=" * 80)
    results, lr, dr = run_sweep(networks, torch_device, args.output_dir,
                                layers_range, devices_range, args.seeds, seeds,
                                versions=versions)

    plot_sweep(results, lr, dr, args.output_dir, versions=versions)
    plot_timing(results, lr, dr, args.output_dir, versions=versions)
    print_gap_table(results, lr, dr, versions=versions)


if __name__ == '__main__':
    main()
