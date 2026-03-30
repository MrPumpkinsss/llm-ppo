"""
evaluate.py - 评估 + 流水线可视化
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
from simulator import generate_llm_layers, generate_heterogeneous_cluster, \
    simulate_inference_tpot, verify_cluster_constraints
from baseline import dp_optimal, greedy_baseline, greedy_memory_aware, \
    uniform_baseline, random_search_baseline


def evaluate_agent(model, env, n_episodes=50, deterministic=True):
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 5000)
        act, _ = model.predict(obs, deterministic=deterministic)
        _, _, _, _, info = env.step(act)
        results.append(info)
    return results


def plot_pipeline(partition, layers, cluster, result, title, ax):
    nd = len(cluster.devices)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    t = 0.0
    blocks = []
    for li in range(len(partition)):
        d = partition[li]
        ct = (layers[li].flops / cluster.devices[d].compute_power) * 1000.0
        blocks.append((li, d, t, t + ct, 'compute'))
        t += ct
        if li < len(partition) - 1 and partition[li + 1] != d:
            d2 = partition[li + 1]
            bw = cluster.bandwidth_matrix[d][d2]
            comm = (layers[li].activation_size / bw) * 1000.0 + 0.3
            blocks.append((li, -1, t, t + comm, 'comm'))
            t += comm
    total_time = t
    for li, d, s, e, typ in blocks:
        if typ == 'comm':
            ax.barh(nd, e-s, left=s, color='#BDBDBD', alpha=0.7,
                    edgecolor='#757575', linewidth=0.5, height=0.5)
            if e-s > total_time * 0.015:
                ax.text((s+e)/2, nd, '↔', ha='center', va='center', fontsize=6)
        else:
            ax.barh(d, e-s, left=s, color=colors[d%10], alpha=0.85,
                    edgecolor='black', linewidth=0.3, height=0.6)
            if e-s > total_time * 0.018:
                ax.text((s+e)/2, d, f'{li}', ha='center', va='center', fontsize=5)
    labels = [f'Dev{i}({cluster.devices[i].compute_power:.0f}G,{cluster.devices[i].memory:.1f}GB)'
              for i in range(nd)] + ['Comm']
    ax.set_yticks(list(range(nd))+[nd])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_xlim(0, total_time*1.02)
    ax.set_title(f'{title}  |  TPOT={result["tpot"]:.1f}ms '
                 f'(comp={result["total_compute_time"]:.1f}+comm={result["total_comm_time"]:.1f}, '
                 f'sw={result["num_device_switches"]})', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='x')


def plot_partition_map(parts_dict, num_layers, ax):
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    methods = list(parts_dict.keys())
    nm = len(methods)
    for mi, (name, part) in enumerate(parts_dict.items()):
        y = nm - mi - 1
        for li, di in enumerate(part):
            di = int(di)
            rect = plt.Rectangle((li, y-0.4), 0.9, 0.8,
                                 facecolor=colors[di%10], edgecolor='black', linewidth=0.3)
            ax.add_patch(rect)
            ax.text(li+0.45, y, str(di), ha='center', va='center', fontsize=5, fontweight='bold')
    ax.set_xlim(-0.5, num_layers+0.5)
    ax.set_ylim(-0.7, nm+0.3)
    ax.set_yticks(range(nm))
    ax.set_yticklabels(list(reversed(methods)), fontsize=8)
    ax.set_xlabel('Layer Index', fontsize=9)
    ax.set_title('Partition Map', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.1, axis='x')
    all_d = sorted(set(int(d) for p in parts_dict.values() for d in p))
    ax.legend(handles=[mpatches.Patch(color=colors[d%10], label=f'Dev{d}') for d in all_d],
              loc='upper right', fontsize=7, ncol=min(6, len(all_d)))


def comprehensive_evaluation(model_path, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    model = PPO.load(model_path, device='cpu')
    print("Model loaded\n")

    test_configs = [
        {'nd': 2, 'seed': 42,  'name': '28L-2D'},
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
        print(f" {tc['name']}  (28L × {tc['nd']}D, seed={tc['seed']})")
        print(f"{'='*65}")

        layers = generate_llm_layers(28)
        cluster = generate_heterogeneous_cluster(tc['nd'], seed=tc['seed'])
        cons = verify_cluster_constraints(layers, cluster)

        for d in cluster.devices:
            print(f"  Dev{d.device_id}: {d.compute_power:6.1f}G, {d.memory:.1f}GB")
        print(f"  Total: {cons['total_params']:.1f}GB | "
              f"{'MUST distribute' if cons['must_distribute'] else 'optional'}")

        env = FixedClusterEnv(tc['nd'], tc['seed'])

        rl_det = evaluate_agent(model, env, 100, True)
        rl_stoch = evaluate_agent(model, env, 200, False)
        all_rl = rl_det + rl_stoch
        rl_best = min(all_rl, key=lambda x: x['tpot'])
        rl_det_tpots = [r['tpot'] for r in rl_det]

        print(f"\n  RL det:  {np.mean(rl_det_tpots):.1f} ± {np.std(rl_det_tpots):.1f}")
        print(f"  RL best: {rl_best['tpot']:.1f}  sw={rl_best['num_device_switches']}")
        print(f"           {rl_best['partition']}")

        dp = dp_optimal(layers, cluster)
        gr = greedy_baseline(layers, cluster)
        gm = greedy_memory_aware(layers, cluster)
        un = uniform_baseline(layers, cluster)
        rs = random_search_baseline(layers, cluster, 50000)

        # 安全打印DP
        if dp is not None:
            print(f"  DP:      {dp['tpot']:.1f}  sw={dp['result']['num_device_switches']}")
            print(f"           {dp['partition']}")
            dp_tpot = dp['tpot']
            dp_part = dp['partition']
            dp_result = dp['result']
        else:
            print(f"  DP:      FAILED (using GrdMem as reference)")
            dp_tpot = gm['tpot']
            dp_part = gm['partition']
            dp_result = gm['result']

        print(f"  Greedy:  {gr['tpot']:.1f} | GrdMem: {gm['tpot']:.1f} | "
              f"Uni: {un['tpot']:.1f} | Rand: {rs['tpot']:.1f}" if rs else
              f"  Greedy:  {gr['tpot']:.1f} | GrdMem: {gm['tpot']:.1f} | "
              f"Uni: {un['tpot']:.1f}")

        gap = (rl_best['tpot'] - dp_tpot) / dp_tpot * 100
        print(f"  Gap RL↔DP: {gap:+.1f}%\n")

        rl_full = simulate_inference_tpot(layers, cluster, rl_best['partition'])

        all_results[tc['name']] = {
            'nd': tc['nd'], 'seed': tc['seed'],
            'rl_mean': float(np.mean(rl_det_tpots)),
            'rl_best': float(rl_best['tpot']),
            'rl_std': float(np.std(rl_det_tpots)),
            'rl_partition': rl_best['partition'],
            'rl_result': rl_full,
            'dp': dp_tpot, 'dp_partition': dp_part, 'dp_result': dp_result,
            'greedy': gr['tpot'], 'greedy_partition': gr['partition'], 'greedy_result': gr['result'],
            'greedy_mem': gm['tpot'], 'gm_partition': gm['partition'], 'gm_result': gm['result'],
            'uniform': un['tpot'],
            'random': rs['tpot'] if rs else None,
            'layers': layers, 'cluster': cluster,
        }

    return all_results


def plot_all(results, metrics, save_dir='results'):
    configs = list(results.keys())
    nc = len(configs)

    # ===== Summary figure =====
    fig1 = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 2, figure=fig1, hspace=0.4, wspace=0.3)

    ax = fig1.add_subplot(gs[0, 0])
    if metrics and metrics.get('timesteps'):
        s = metrics['timesteps']
        ax.plot(s, metrics['mean_tpot'], 'b-', lw=2, label='Mean')
        ax.plot(s, metrics['min_tpot'], 'g--', lw=2, label='Min')
        m = np.array(metrics['mean_tpot']); sd = np.array(metrics['std_tpot'])
        ax.fill_between(s, m-sd, m+sd, alpha=0.15, color='blue')
    ax.set_xlabel('Steps'); ax.set_ylabel('TPOT (ms)')
    ax.set_title('Training TPOT', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig1.add_subplot(gs[0, 1])
    if metrics and metrics.get('timesteps'):
        ax.plot(metrics['timesteps'], metrics['mean_reward'], 'r-', lw=2, label='Reward')
        ax2 = ax.twinx()
        ax2.plot(metrics['timesteps'], metrics['ent_coef'], 'purple', lw=1.5, alpha=0.5, label='Entropy')
        ax2.set_ylabel('Ent', color='purple'); ax2.legend(loc='lower right')
    ax.set_xlabel('Steps'); ax.set_ylabel('Reward')
    ax.set_title('Reward & Entropy', fontweight='bold'); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)

    ax = fig1.add_subplot(gs[1, :])
    x = np.arange(nc); w = 0.12
    for mi, (meth, col, key) in enumerate([
        ('RL best', '#2196F3', 'rl_best'), ('RL mean', '#90CAF9', 'rl_mean'),
        ('DP', '#4CAF50', 'dp'), ('Greedy', '#FF9800', 'greedy'),
        ('GrdMem', '#FFB74D', 'greedy_mem'), ('Uniform', '#F44336', 'uniform'),
        ('Rand50k', '#9C27B0', 'random')
    ]):
        vals = [results[c].get(key) or 0 for c in configs]
        bars = ax.bar(x+mi*w-3*w, vals, w, label=meth, color=col, alpha=0.85)
        for b, v in zip(bars, vals):
            if v > 0: ax.text(b.get_x()+w/2, b.get_height(), f'{v:.0f}',
                              ha='center', va='bottom', fontsize=5, rotation=55)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel('TPOT (ms)'); ax.set_title('All Methods', fontweight='bold')
    ax.legend(fontsize=7, ncol=4); ax.grid(True, alpha=0.3, axis='y')

    ax = fig1.add_subplot(gs[2, 0])
    for mi, (meth, col, key) in enumerate([
        ('RL', '#2196F3', 'rl_best'), ('Greedy', '#FF9800', 'greedy'),
        ('GrdMem', '#FFB74D', 'greedy_mem'), ('Uniform', '#F44336', 'uniform')
    ]):
        gaps = [(results[c][key] - results[c]['dp']) / results[c]['dp'] * 100 for c in configs]
        ax.bar(np.arange(nc)+mi*0.2-0.3, gaps, 0.2, label=meth, color=col, alpha=0.85)
    ax.axhline(0, color='green', ls='--', lw=2)
    ax.set_xticks(np.arange(nc)); ax.set_xticklabels(configs, fontsize=7, rotation=20)
    ax.set_ylabel('Gap from DP (%)'); ax.set_title('Optimality Gap', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = fig1.add_subplot(gs[2, 1])
    sc = [c for c in configs if results[c]['seed'] == 42]
    nds = [results[c]['nd'] for c in sc]
    for key, marker, label in [('rl_best','bo-','RL'),('dp','g^-','DP'),
                                 ('greedy','rs-','Greedy'),('greedy_mem','mD-','GrdMem')]:
        ax.plot(nds, [results[c][key] for c in sc], marker, lw=2, ms=8, label=label)
    ax.set_xlabel('Devices'); ax.set_ylabel('TPOT')
    ax.set_title('Scalability (seed=42)', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    fig1.savefig(os.path.join(save_dir, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("Saved summary.png")

    # ===== Pipeline per config =====
    for cfg in configs:
        r = results[cfg]
        layers, cluster = r['layers'], r['cluster']

        fig, axes = plt.subplots(5, 1, figsize=(24, 20),
                                 gridspec_kw={'height_ratios': [1,1,1,1,0.7]})
        fig.suptitle(f'{cfg}: Pipeline Comparison', fontsize=14, fontweight='bold', y=0.995)

        plot_pipeline(r['rl_partition'], layers, cluster, r['rl_result'], 'RL Agent', axes[0])
        plot_pipeline(r['dp_partition'], layers, cluster, r['dp_result'], 'DP Optimal', axes[1])
        plot_pipeline(r['gm_partition'], layers, cluster, r['gm_result'], 'Greedy Memory-Aware', axes[2])
        plot_pipeline(r['greedy_partition'], layers, cluster, r['greedy_result'], 'Greedy Compute', axes[3])

        parts = {
            f"DP ({r['dp']:.0f}ms)": r['dp_partition'],
            f"RL ({r['rl_best']:.0f}ms)": r['rl_partition'],
            f"GrdMem ({r['greedy_mem']:.0f}ms)": r['gm_partition'],
            f"Greedy ({r['greedy']:.0f}ms)": r['greedy_partition'],
        }
        plot_partition_map(parts, 28, axes[4])

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fname = f'pipeline_{cfg.replace("-","_")}.png'
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")


def print_summary(results):
    print("\n" + "=" * 120)
    print(f"{'Config':<14} {'RL Best':>8} {'RL Mean':>8} {'DP':>8} {'Greedy':>8} "
          f"{'GrdMem':>8} {'Uniform':>8} {'Rand50k':>8} {'RL↔DP':>8}")
    print("-" * 120)
    for cfg, r in results.items():
        gap = (r['rl_best'] - r['dp']) / r['dp'] * 100
        rs = f"{r['random']:.1f}" if r.get('random') else "N/A"
        print(f"{cfg:<14} {r['rl_best']:>8.1f} {r['rl_mean']:>8.1f} {r['dp']:>8.1f} "
              f"{r['greedy']:>8.1f} {r['greedy_mem']:>8.1f} {r['uniform']:>8.1f} "
              f"{rs:>8} {gap:>+7.1f}%")
    print("=" * 120)
    v = list(results.values())
    bg = sum(1 for r in v if r['rl_best'] < r['greedy'])
    bgm = sum(1 for r in v if r['rl_best'] < r['greedy_mem'])
    bu = sum(1 for r in v if r['rl_best'] < r['uniform'])
    avg = np.mean([(r['rl_best'] - r['dp']) / r['dp'] * 100 for r in v])
    print(f"RL < Greedy: {bg}/{len(v)} | RL < GrdMem: {bgm}/{len(v)} | "
          f"RL < Uniform: {bu}/{len(v)} | Avg gap: {avg:+.1f}%")