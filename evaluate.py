"""
evaluate.py - 28层专用评估 + 流水线可视化
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
from env import FixedClusterEnv, MAX_DEVICES
from simulator import generate_llm_layers, generate_heterogeneous_cluster, verify_cluster_constraints
from baseline import dp_optimal, greedy_baseline, greedy_memory_aware, uniform_baseline, random_search_baseline


def evaluate_agent(model, env, n_episodes=50, deterministic=True):
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 5000)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
        
        # Now the episode is finished → info contains 'tpot'
        if 'tpot' in info:
            results.append(info)
        else:
            # Fallback (should not happen)
            print(f"Warning: No tpot in info for episode {ep}")
    return results


def plot_pipeline(partition, layers, cluster, result, title, ax):
    """画单个分配方案的流水线时间线"""
    nd = len(cluster.devices)
    dev_colors = plt.cm.Set2(np.linspace(0, 1, max(8, nd)))

    # 每设备上的层执行时间线
    dev_start = {}   # dev -> 当前结束时间
    layer_times = [] # (layer_idx, dev, start, end, is_comm)

    current_time = 0.0
    for li in range(len(partition)):
        d = partition[li]
        ct = (layers[li].flops / cluster.devices[d].compute_power) * 1000.0

        layer_times.append((li, d, current_time, current_time + ct, False))
        current_time += ct

        # 通信
        if li < len(partition) - 1 and partition[li + 1] != d:
            d2 = partition[li + 1]
            bw = cluster.bandwidth_matrix[d][d2]
            comm_t = (layers[li].activation_size / bw) * 1000.0 + 0.3
            layer_times.append((li, -1, current_time, current_time + comm_t, True))
            current_time += comm_t

    # 画图
    for li, d, start, end, is_comm in layer_times:
        if is_comm:
            ax.barh(nd, end - start, left=start, color='gray', alpha=0.5,
                    edgecolor='black', linewidth=0.3, height=0.6)
        else:
            ax.barh(d, end - start, left=start, color=dev_colors[d], alpha=0.8,
                    edgecolor='black', linewidth=0.3, height=0.6)
            if end - start > current_time * 0.02:
                ax.text((start + end) / 2, d, f'L{li}', ha='center', va='center',
                        fontsize=5, fontweight='bold')

    dev_labels = [f'Dev{i} ({cluster.devices[i].compute_power:.0f}G, {cluster.devices[i].memory:.1f}GB)'
                  for i in range(nd)]
    dev_labels.append('Comm')

    ax.set_yticks(list(range(nd)) + [nd])
    ax.set_yticklabels(dev_labels, fontsize=8)
    ax.set_xlabel('Time (ms)', fontsize=9)
    ax.set_title(f'{title}\nTPOT={result["tpot"]:.1f}ms  '
                 f'(compute={result["total_compute_time"]:.1f} + comm={result["total_comm_time"]:.1f})',
                 fontsize=10, fontweight='bold')
    ax.set_xlim(0, current_time * 1.05)
    ax.grid(True, alpha=0.2, axis='x')


def plot_partition_map(partitions_dict, num_layers, ax):
    """画分配方案对比图（每层→设备的颜色块）"""
    dev_colors = plt.cm.Set2(np.linspace(0, 1, 10))
    methods = list(partitions_dict.keys())
    nm = len(methods)

    for mi, (name, part) in enumerate(partitions_dict.items()):
        y = nm - mi - 1
        for li, di in enumerate(part):
            di = int(di)
            rect = plt.Rectangle((li, y - 0.4), 0.9, 0.8,
                                 facecolor=dev_colors[di], edgecolor='black', linewidth=0.3)
            ax.add_patch(rect)
            ax.text(li + 0.45, y, str(di), ha='center', va='center', fontsize=6, fontweight='bold')

    ax.set_xlim(-0.5, num_layers + 0.5)
    ax.set_ylim(-0.8, nm + 0.3)
    ax.set_yticks(range(nm))
    ax.set_yticklabels(list(reversed(methods)), fontsize=9)
    ax.set_xlabel('Layer Index', fontsize=10)
    ax.set_title('Partition Comparison (number = device ID)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='x')

    all_devs = sorted(set(int(d) for p in partitions_dict.values() for d in p))
    ax.legend(handles=[mpatches.Patch(color=dev_colors[d], label=f'Dev {d}') for d in all_devs],
              loc='upper right', fontsize=8, ncol=len(all_devs))


def comprehensive_evaluation(model_path, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    model = PPO.load(model_path, device='cpu')
    print("Model loaded\n")

    test_configs = [
       # {'nd': 2, 'seed': 42,  'name': '28L-2D'},
        {'nd': 3, 'seed': 42,  'name': '28L-3D'},
        {'nd': 3, 'seed': 123, 'name': '28L-3D-s2'},
        {'nd': 4, 'seed': 42,  'name': '28L-4D'},
        {'nd': 4, 'seed': 789, 'name': '28L-4D-s2'},
        {'nd': 5, 'seed': 42,  'name': '28L-5D'},
        {'nd': 5, 'seed': 456, 'name': '28L-5D-s2'},
        {'nd': 6, 'seed': 42,  'name': '28L-6D'},
    ]

    all_results = {}

    for tc in test_configs:
        print(f"{'='*65}")
        print(f" {tc['name']}  (28 layers × {tc['nd']} devices, seed={tc['seed']})")
        print(f"{'='*65}")

        layers = generate_llm_layers(28)
        cluster = generate_heterogeneous_cluster(tc['nd'], seed=tc['seed'])
        cons = verify_cluster_constraints(layers, cluster)

        for d in cluster.devices:
            print(f"  Dev{d.device_id}: {d.compute_power:6.1f} GFLOPS, {d.memory:.1f} GB")
        print(f"  Total params: {cons['total_params']:.1f} GB | Distribute: "
              f"{'MUST ✓' if cons['must_distribute'] else 'optional'}")

        env = FixedClusterEnv(tc['nd'], tc['seed'])
        rl_det = evaluate_agent(model, env, 80, True)
        rl_stoch = evaluate_agent(model, env, 150, False)
        all_rl = rl_det + rl_stoch
        rl_best = min(all_rl, key=lambda x: x['tpot'])
        rl_det_tpots = [r['tpot'] for r in rl_det]

        print(f"\n  RL det:  {np.mean(rl_det_tpots):.1f} ± {np.std(rl_det_tpots):.1f} ms")
        print(f"  RL best: {rl_best['tpot']:.1f} ms  {rl_best['partition']}")

        dp = dp_optimal(layers, cluster)
        print(f"  DP:      {dp['tpot']:.1f} ms  {dp['partition']}")

        gr = greedy_baseline(layers, cluster)
        gm = greedy_memory_aware(layers, cluster)
        un = uniform_baseline(layers, cluster)
        rs = random_search_baseline(layers, cluster, 50000)

        print(f"  Greedy:  {gr['tpot']:.1f} ms | GreedyMem: {gm['tpot']:.1f} ms | "
              f"Uniform: {un['tpot']:.1f} ms | Rand50k: {rs['tpot']:.1f}" if rs else "")

        opt = dp['tpot']
        gap = (rl_best['tpot'] - opt) / opt * 100
        print(f"  RL gap from DP: {gap:+.1f}%\n")

        all_results[tc['name']] = {
            'nd': tc['nd'], 'seed': tc['seed'],
            'rl_mean': float(np.mean(rl_det_tpots)),
            'rl_best': float(rl_best['tpot']),
            'rl_std': float(np.std(rl_det_tpots)),
            'rl_partition': rl_best['partition'],
            'rl_result': rl_best,
            'dp': dp['tpot'], 'dp_partition': dp['partition'], 'dp_result': dp['result'],
            'greedy': gr['tpot'], 'greedy_partition': gr['partition'], 'greedy_result': gr['result'],
            'greedy_mem': gm['tpot'], 'greedy_mem_partition': gm['partition'],
            'uniform': un['tpot'],
            'random': rs['tpot'] if rs else None,
            'layers': layers, 'cluster': cluster,
        }

    return all_results


def plot_all(results, metrics, save_dir='results'):
    configs = list(results.keys())
    n_cfgs = len(configs)

    # ==========================================
    # Figure 1: 训练曲线 + 柱状对比 + gap
    # ==========================================
    fig1 = plt.figure(figsize=(20, 16))
    gs1 = GridSpec(3, 2, figure=fig1, hspace=0.35, wspace=0.3)

    # 训练TPOT
    ax = fig1.add_subplot(gs1[0, 0])
    if metrics and metrics.get('timesteps'):
        s = metrics['timesteps']
        ax.plot(s, metrics['mean_tpot'], 'b-', lw=2, label='Mean TPOT')
        ax.plot(s, metrics['min_tpot'], 'g--', lw=2, label='Min TPOT')
        m, sd = np.array(metrics['mean_tpot']), np.array(metrics['std_tpot'])
        ax.fill_between(s, m - sd, m + sd, alpha=0.15, color='blue')
    ax.set_xlabel('Steps'); ax.set_ylabel('TPOT (ms)')
    ax.set_title('Training TPOT', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    # 训练Reward + Entropy
    ax = fig1.add_subplot(gs1[0, 1])
    if metrics and metrics.get('timesteps'):
        ax.plot(metrics['timesteps'], metrics['mean_reward'], 'r-', lw=2, label='Reward')
        ax2 = ax.twinx()
        ax2.plot(metrics['timesteps'], metrics['ent_coef'], 'purple', lw=1.5, alpha=0.6, label='Entropy')
        ax2.set_ylabel('Entropy Coef', color='purple'); ax2.legend(loc='lower right')
    ax.set_xlabel('Steps'); ax.set_ylabel('Reward')
    ax.set_title('Reward & Entropy', fontweight='bold'); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)

    # 柱状对比
    ax = fig1.add_subplot(gs1[1, :])
    x = np.arange(n_cfgs); w = 0.13
    for mi, (meth, col) in enumerate([
        ('RL best', '#2196F3'), ('RL mean', '#90CAF9'), ('DP', '#4CAF50'),
        ('Greedy', '#FF9800'), ('GreedyMem', '#FFB74D'), ('Uniform', '#F44336'), ('Rand50k', '#9C27B0')
    ]):
        vals = []
        for cfg in configs:
            r = results[cfg]
            if meth == 'RL best': vals.append(r['rl_best'])
            elif meth == 'RL mean': vals.append(r['rl_mean'])
            elif meth == 'DP': vals.append(r['dp'])
            elif meth == 'Greedy': vals.append(r['greedy'])
            elif meth == 'GreedyMem': vals.append(r['greedy_mem'])
            elif meth == 'Uniform': vals.append(r['uniform'])
            elif meth == 'Rand50k': vals.append(r.get('random') or 0)
        bars = ax.bar(x + mi * w - 3 * w, vals, w, label=meth, color=col, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + w / 2, bar.get_height(), f'{val:.0f}',
                        ha='center', va='bottom', fontsize=5, rotation=55)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel('TPOT (ms)'); ax.set_title('TPOT: All Methods × All Configs', fontweight='bold')
    ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.3, axis='y')

    # Gap 图
    ax = fig1.add_subplot(gs1[2, 0])
    gap_m = {'RL best': '#2196F3', 'Greedy': '#FF9800', 'Uniform': '#F44336'}
    bw = 0.25
    for mi, (meth, col) in enumerate(gap_m.items()):
        gaps = []
        for cfg in configs:
            r = results[cfg]
            opt = r['dp']
            if meth == 'RL best': v = r['rl_best']
            elif meth == 'Greedy': v = r['greedy']
            else: v = r['uniform']
            gaps.append((v - opt) / opt * 100)
        ax.bar(np.arange(n_cfgs) + mi * bw - bw, gaps, bw, label=meth, color=col, alpha=0.85)
    ax.axhline(0, color='green', ls='--', lw=2, label='DP optimal')
    ax.set_xticks(np.arange(n_cfgs)); ax.set_xticklabels(configs, fontsize=7, rotation=20)
    ax.set_ylabel('Gap from DP (%)'); ax.set_title('Optimality Gap', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # Scalability
    ax = fig1.add_subplot(gs1[2, 1])
    # 按设备数聚合（只取seed=42的）
    scale_cfgs = [c for c in configs if results[c]['seed'] == 42]
    nds = [results[c]['nd'] for c in scale_cfgs]
    ax.plot(nds, [results[c]['rl_best'] for c in scale_cfgs], 'bo-', lw=2, ms=8, label='RL')
    ax.plot(nds, [results[c]['dp'] for c in scale_cfgs], 'g^-', lw=2, ms=8, label='DP')
    ax.plot(nds, [results[c]['greedy'] for c in scale_cfgs], 'rs-', lw=2, ms=8, label='Greedy')
    ax.set_xlabel('Number of Devices'); ax.set_ylabel('TPOT (ms)')
    ax.set_title('Scalability (seed=42)', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    fig1.savefig(os.path.join(save_dir, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved summary.png")

    # ==========================================
    # Figure 2-N: 每个配置的流水线 + 分配图
    # ==========================================
    for cfg_name in configs:
        r = results[cfg_name]
        layers, cluster = r['layers'], r['cluster']
        nd = r['nd']

        fig, axes = plt.subplots(4, 1, figsize=(20, 16),
                                  gridspec_kw={'height_ratios': [1, 1, 1, 0.8]})
        fig.suptitle(f'{cfg_name}: Pipeline Timeline Comparison', fontsize=14, fontweight='bold', y=0.98)

        # RL流水线
        plot_pipeline(r['rl_partition'], layers, cluster, r['rl_result'],
                      f'RL Agent (best)', axes[0])

        # DP流水线
        plot_pipeline(r['dp_partition'], layers, cluster, r['dp_result'],
                      f'DP Optimal', axes[1])

        # Greedy流水线
        plot_pipeline(r['greedy_partition'], layers, cluster, r['greedy_result'],
                      f'Greedy (compute-proportional)', axes[2])

        # 分配方案对比
        parts = {}
        parts[f"DP ({r['dp']:.0f}ms)"] = r['dp_partition']
        parts[f"RL ({r['rl_best']:.0f}ms)"] = r['rl_partition']
        parts[f"Greedy ({r['greedy']:.0f}ms)"] = r['greedy_partition']
        plot_partition_map(parts, 28, axes[3])

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f'pipeline_{cfg_name.replace("-","_")}.png'
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")


def print_summary(results):
    print("\n" + "=" * 115)
    print(f"{'Config':<14} {'RL Best':>8} {'RL Mean':>8} {'DP':>8} {'Greedy':>8} "
          f"{'GrdMem':>8} {'Uniform':>8} {'Rand50k':>8} {'RL↔DP':>8}")
    print("-" * 115)
    for cfg, r in results.items():
        gap = (r['rl_best'] - r['dp']) / r['dp'] * 100
        rs = f"{r['random']:.1f}" if r.get('random') else "N/A"
        print(f"{cfg:<14} {r['rl_best']:>8.1f} {r['rl_mean']:>8.1f} {r['dp']:>8.1f} "
              f"{r['greedy']:>8.1f} {r['greedy_mem']:>8.1f} {r['uniform']:>8.1f} "
              f"{rs:>8} {gap:>+7.1f}%")
    print("=" * 115)
    valid = list(results.values())
    bg = sum(1 for r in valid if r['rl_best'] < r['greedy'])
    bu = sum(1 for r in valid if r['rl_best'] < r['uniform'])
    avg_gap = np.mean([(r['rl_best'] - r['dp']) / r['dp'] * 100 for r in valid])
    print(f"RL beats Greedy: {bg}/{len(valid)} | RL beats Uniform: {bu}/{len(valid)} | Avg gap from DP: {avg_gap:+.1f}%")