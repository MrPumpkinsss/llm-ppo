"""
main.py - 支持训练/加载已有模型
用法:
  python main.py                  # 训练新模型
  python main.py --load results/ppo_llm_partition   # 加载已有模型
"""

import os, sys, json, argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='LLM Layer Partition via RL')
    parser.add_argument('--load', type=str, default=None,
                        help='加载已训练模型路径 (不含.zip后缀)')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='训练总步数')
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("  LLM Layer Partition via RL")
    print("  28 layers × 2~6 devices")
    print("=" * 70)

    # ============ 训练 or 加载 ============
    if args.load:
        print(f"\n📦 Loading model from: {args.load}")
        from stable_baselines3 import PPO
        model = PPO.load(args.load, device='cpu')
        print("  ✅ Model loaded")

        metrics_path = os.path.join(os.path.dirname(args.load), 'training_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                training_metrics = json.load(f)
            print(f"  ✅ Training metrics loaded")
        else:
            training_metrics = {}
            print(f"  ⚠️ No training metrics found")
        model_path = args.load
    else:
        print(f"\n🏋️ Training new model ({args.timesteps} steps)...")
        from train import train
        config = {
            'total_timesteps': args.timesteps,
            'n_envs': 16, 'learning_rate': 3e-4,
            'n_steps': 64, 'batch_size': 32, 'n_epochs': 20,
            'gamma': 1.0, 'gae_lambda': 0.95, 'ent_coef': 0.2,
            'clip_range': 0.2, 'eval_freq': 500, 'save_dir': save_dir,
        }
        model, training_metrics, config = train(config)
        model_path = os.path.join(save_dir, 'ppo_llm_partition')

    # ============ 评估 ============
    print("\n" + "=" * 60)
    print(" Evaluation: 28 layers × {2,3,4,5,6} devices")
    print("=" * 60)

    from evaluate import comprehensive_evaluation, plot_all, print_summary

    results = comprehensive_evaluation(model_path, save_dir)

    # ============ 画图 ============
    print("\n" + "=" * 60)
    print(" Generating plots...")
    print("=" * 60)

    plot_all(results, training_metrics, save_dir)
    print_summary(results)

    # ============ 保存数据 ============
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer))
                                else vv.tolist() if isinstance(vv, np.ndarray)
                                else vv)
                           for kk, vv in v.items()
                           if kk not in ('layers', 'cluster', 'rl_result', 'dp_result', 'greedy_result')}
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n📁 All outputs in: {save_dir}/")
    print(f"   summary.png          - 训练曲线 + 柱状对比 + gap + scalability")
    print(f"   pipeline_*.png       - 每个配置的流水线时间线")
    print(f"   results.json         - 数值结果")
    print(f"\n{'='*70}")
    print(" DONE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()