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

from config import TrainConfig
from environment import create_random_config, compute_simple_tpot
from baselines import dp_partition, brute_force_optimal, beam_search_dp
from agents.dqn_v1 import DQNv1Network, dqn_v1_inference
from agents.ppo_v2 import PPOv2Network, ppo_v2_inference
from agents.ppo_v3 import PPOv3Network, ppo_v3_inference
from agents.ppo_v4 import PPOv4Network, ppo_v4_inference
from agents.maskable_ppo_v5 import MaskablePPOv5Network, maskable_v5_inference
from agents.gnn_ppo_v6 import PPOv6Network, ppo_v6_inference
from agents.gnn_ar_ppo_v7 import PPOv7Network, ppo_v7_inference


def load_all_models(model_dir='results'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_device = torch.device(device)
    networks = {}

    for ver, NetClass in [('v1', DQNv1Network), ('v2', PPOv2Network),
                           ('v3', PPOv3Network), ('v4', PPOv4Network),
                           ('v5', MaskablePPOv5Network), ('v6', PPOv6Network),
                           ('v7', PPOv7Network)]:
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
              layers_range=None, devices_range=None, runs=10, seeds=None):
    if layers_range is None:
        layers_range = list(range(24, 65, 4))
    if devices_range is None:
        devices_range = [4, 6, 8, 10]
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024, 1337, 2048, 3141, 4096, 5555,
                 6789, 7777, 8888, 9999, 10101]

    results = {nd: {k: [] for k in ['layers', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'dp', 'brute',
                                      't_v1', 't_v2', 't_v3', 't_v4', 't_v5', 't_v6', 't_v7', 't_dp', 't_brute']}
               for nd in devices_range}

    for nd in devices_range:
        do_brute = True  # always compute best-known: brute for ≤4, top-k for >4

        for nl in layers_range:
            tpot_v1, tpot_v2, tpot_v3, tpot_v4, tpot_v5, tpot_v6, tpot_v7 = [], [], [], [], [], [], []
            tpot_dp_list, tpot_brute_list = [], []
            t_v1, t_v2, t_v3, t_v4, t_v5, t_v6, t_v7, t_dp, t_brute = [], [], [], [], [], [], [], [], []

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

                # V1
                if 'v1' in networks:
                    t0 = time.time()
                    _, tpot = dqn_v1_inference(networks['v1'], devs, lys, ts, nl, nd, torch_device)
                    tpot_v1.append(tpot)
                    t_v1.append(time.time() - t0)
                # V2
                if 'v2' in networks:
                    t0 = time.time()
                    _, tpot = ppo_v2_inference(networks['v2'], devs, lys, ts, nl, nd, torch_device)
                    tpot_v2.append(tpot)
                    t_v2.append(time.time() - t0)
                # V3
                if 'v3' in networks:
                    t0 = time.time()
                    _, tpot = ppo_v3_inference(networks['v3'], devs, lys, ts, nl, nd, torch_device)
                    tpot_v3.append(tpot)
                    t_v3.append(time.time() - t0)
                # V4
                if 'v4' in networks:
                    t0 = time.time()
                    _, tpot = ppo_v4_inference(networks['v4'], devs, lys, ts, nl, nd, torch_device)
                    tpot_v4.append(tpot)
                    t_v4.append(time.time() - t0)
                # V5
                if 'v5' in networks:
                    t0 = time.time()
                    _, tpot = maskable_v5_inference(networks['v5'], devs, lys, ts, nl, nd, torch_device)
                    tpot_v5.append(tpot)
                    t_v5.append(time.time() - t0)
                # V6
                if 'v6' in networks:
                    t0 = time.time()
                    _, tpot = ppo_v6_inference(networks['v6'], devs, lys, ts, nl, nd, torch_device)
                    tpot_v6.append(tpot)
                    t_v6.append(time.time() - t0)
                # V7
                if 'v7' in networks:
                    t0 = time.time()
                    _, tpot = ppo_v7_inference(networks['v7'], devs, lys, ts, nl, nd, torch_device)
                    tpot_v7.append(tpot)
                    t_v7.append(time.time() - t0)

            rec = results[nd]
            rec['layers'].append(nl)
            rec['v1'].append(np.mean(tpot_v1))
            rec['v2'].append(np.mean(tpot_v2))
            rec['v3'].append(np.mean(tpot_v3))
            rec['v4'].append(np.mean(tpot_v4))
            rec['v5'].append(np.mean(tpot_v5))
            rec['v6'].append(np.mean(tpot_v6))
            rec['v7'].append(np.mean(tpot_v7))
            rec['dp'].append(np.mean(tpot_dp_list))
            rec['t_v1'].append(np.mean(t_v1) * 1000)
            rec['t_v2'].append(np.mean(t_v2) * 1000)
            rec['t_v3'].append(np.mean(t_v3) * 1000)
            rec['t_v4'].append(np.mean(t_v4) * 1000)
            rec['t_v5'].append(np.mean(t_v5) * 1000)
            rec['t_v6'].append(np.mean(t_v6) * 1000)
            rec['t_v7'].append(np.mean(t_v7) * 1000)
            rec['t_dp'].append(np.mean(t_dp) * 1000)

            if do_brute:
                rec['brute'].append(np.mean(tpot_brute_list))
                rec['t_brute'].append(np.mean(t_brute) * 1000)
            else:
                rec['brute'].append(None)
                rec['t_brute'].append(None)

            brute_str = f" Brute={np.mean(tpot_brute_list):.3f}" if do_brute else ""
            print(f"  {nd}D/{nl}L | v1={np.mean(tpot_v1):.3f} v2={np.mean(tpot_v2):.3f} "
                  f"v3={np.mean(tpot_v3):.3f} v4={np.mean(tpot_v4):.3f} v5={np.mean(tpot_v5):.3f} "
                  f"v6={np.mean(tpot_v6):.3f} v7={np.mean(tpot_v7):.3f} "
                  f"DP={np.mean(tpot_dp_list):.3f}{brute_str}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'scaling_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results, layers_range, devices_range


def plot_sweep(results, layers_range, devices_range, output_dir):
    fig, axes = plt.subplots(2, len(devices_range), figsize=(5 * len(devices_range), 10))
    if len(devices_range) == 1:
        axes = axes.reshape(2, 1)

    colors = {'v1': '#3b82f6', 'v2': '#22c55e', 'v3': '#f97316', 'v4': '#a855f7',
              'v5': '#06b6d4', 'v6': '#e91e63', 'v7': '#8b5cf6', 'dp': '#f59e0b', 'brute': '#dc2626'}
    markers = {'v1': 'o', 'v2': 's', 'v3': 'D', 'v4': 'P', 'v5': 'X', 'v6': 'v', 'v7': 'H', 'dp': '^', 'brute': '*'}

    # Row 0: Gap to brute-force or DP (%)
    for col, nd in enumerate(devices_range):
        ax = axes[0, col]
        r = results[nd]
        ls = r['layers']
        dp = np.array(r['dp'])

        # Use brute-force as baseline if available, otherwise DP
        baseline_key = 'brute' if r['brute'][0] is not None else 'dp'
        baseline = np.array([v if v is not None else dp[i] for i, v in enumerate(r['brute'])])

        for v, label in [('v1', 'V1-DQN'), ('v2', 'V2-PPO'), ('v3', 'V3-PPO'),
                         ('v4', 'V4-PPO'), ('v5', 'V5-MaskPPO'), ('v6', 'V6-GNN-PPO'),
                         ('v7', 'V7-AR-GNN')]:
            gap = (np.array(r[v]) - baseline) / baseline * 100
            ax.plot(ls, gap, marker=markers[v], color=colors[v],
                    label=label, linewidth=2, markersize=5)

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
        for v, label in [('v1', 'V1-DQN'), ('v2', 'V2-PPO'), ('v3', 'V3-PPO'),
                         ('v4', 'V4-PPO'), ('v5', 'V5-MaskPPO'), ('v6', 'V6-GNN-PPO'),
                         ('v7', 'V7-AR-GNN')]:
            ax.plot(ls, r[v], marker=markers[v], color=colors[v],
                    label=label, linewidth=2, markersize=5)

        ax.set_xlabel('Layers')
        ax.set_ylabel('Avg TPOT')
        ax.set_title(f'{nd} Devices — Raw TPOT')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('V1-V7 vs DP Scaling\n(avg of 10 runs)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, 'scaling_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


def plot_timing(results, layers_range, devices_range, output_dir):
    """Separate timing plot: inference time vs layers for each algorithm."""
    fig, axes = plt.subplots(1, len(devices_range), figsize=(5 * len(devices_range), 5))
    if len(devices_range) == 1:
        axes = [axes]

    colors = {'v1': '#3b82f6', 'v2': '#22c55e', 'v3': '#f97316', 'v4': '#a855f7',
              'v5': '#06b6d4', 'v6': '#e91e63', 'v7': '#8b5cf6', 'dp': '#f59e0b', 'brute': '#dc2626'}
    markers = {'v1': 'o', 'v2': 's', 'v3': 'D', 'v4': 'P', 'v5': 'X', 'v6': 'v', 'v7': 'H', 'dp': '^', 'brute': '*'}

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
        for v, label in [('v1', 'V1-DQN'), ('v2', 'V2-PPO'), ('v3', 'V3-PPO'),
                         ('v4', 'V4-PPO'), ('v5', 'V5-MaskPPO'), ('v6', 'V6-GNN-PPO'),
                         ('v7', 'V7-AR-GNN')]:
            ax.plot(ls, r[f't_{v}'], marker=markers[v], color=colors[v],
                    label=label, linewidth=2, markersize=5)

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


def print_gap_table(results, layers_range, devices_range):
    print("\n" + "=" * 170)
    print("GAP TO DP (%)" + ("  — Brute-force gap shown as DP gap" if any(results[nd]['brute'][0] is not None for nd in devices_range) else ""))
    print(f"{'Layers':>8}", end='')
    for nd in devices_range:
        has_brute = results[nd]['brute'][0] is not None
        brute_col = f" {nd}D-DP " if has_brute else ""
        print(f"  {nd}D-v1  {nd}D-v2  {nd}D-v3  {nd}D-v4  {nd}D-v5  {nd}D-v6  {nd}D-v7{brute_col}", end='')
    print()
    for li, nl in enumerate(layers_range):
        print(f"{nl:>8}", end='')
        for nd in devices_range:
            r = results[nd]
            dp = r['dp'][li]
            v1g = (r['v1'][li] - dp) / dp * 100
            v2g = (r['v2'][li] - dp) / dp * 100
            v3g = (r['v3'][li] - dp) / dp * 100
            v4g = (r['v4'][li] - dp) / dp * 100
            v5g = (r['v5'][li] - dp) / dp * 100
            v6g = (r['v6'][li] - dp) / dp * 100
            v7g = (r['v7'][li] - dp) / dp * 100
            end = ""
            if r['brute'][li] is not None:
                dpg = (dp - r['brute'][li]) / r['brute'][li] * 100
                end = f" {dpg:+6.2f}"
            print(f"  {v1g:+6.2f} {v2g:+6.2f} {v3g:+6.2f} {v4g:+6.2f} {v5g:+6.2f} {v6g:+6.2f} {v7g:+6.2f}{end}", end='')
        print()


def main():
    parser = argparse.ArgumentParser(description='Scaling Sweep')
    parser.add_argument('--model-dir', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--seeds', type=int, default=15)
    args = parser.parse_args()

    print("Loading models...")
    networks, torch_device = load_all_models(args.model_dir)

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
                                layers_range, devices_range, args.seeds, seeds)

    plot_sweep(results, lr, dr, args.output_dir)
    plot_timing(results, lr, dr, args.output_dir)
    print_gap_table(results, lr, dr)


if __name__ == '__main__':
    main()
