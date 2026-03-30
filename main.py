"""
main.py
用法:
  python main.py                                    # 训练新模型
  python main.py --load results/ppo_llm_partition   # 加载已有模型
  python main.py --timesteps 2000000                # 更多训练步
"""

import os, json, argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None, help='模型路径(不含.zip)')
    parser.add_argument('--timesteps', type=int, default=1000_000)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("  LLM 28-Layer Partition via RL")
    print("  Action: cut-points + device assignment (continuous)")
    print("=" * 70)

    if args.load:
        print(f"\n📦 Loading: {args.load}")
        from stable_baselines3 import PPO
        model = PPO.load(args.load, device='cpu')
        print("  ✅ Loaded")
        mp_dir = os.path.dirname(args.load) or save_dir
        mf = os.path.join(mp_dir, 'training_metrics.json')
        metrics = json.load(open(mf)) if os.path.exists(mf) else {}
        model_path = args.load
    else:
        print(f"\n🏋️ Training ({args.timesteps} steps)...")
        from train import train
        config = {
            'total_timesteps': args.timesteps,
            'n_envs': 16, 'learning_rate': 3e-4,
            'n_steps': 128, 'batch_size': 64, 'n_epochs': 15,
            'gamma': 1.0, 'gae_lambda': 0.95, 'ent_coef': 0.05,
            'clip_range': 0.2, 'eval_freq': 500, 'save_dir': save_dir,
        }
        model, metrics, config = train(config)
        model_path = os.path.join(save_dir, 'ppo_llm_partition')

    print("\n" + "=" * 60)
    print(" Evaluating 28L × {2,3,4,5,6}D...")
    print("=" * 60)

    from evaluate import comprehensive_evaluation, plot_all, print_summary
    results = comprehensive_evaluation(model_path, save_dir)

    print("\n" + "=" * 60)
    print(" Plotting...")
    print("=" * 60)
    plot_all(results, metrics, save_dir)
    print_summary(results)

    # Save JSON
    ser = {}
    for k, v in results.items():
        ser[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer))
                       else vv.tolist() if isinstance(vv, np.ndarray) else vv)
                  for kk, vv in v.items()
                  if kk not in ('layers', 'cluster', 'rl_result', 'dp_result',
                                'greedy_result', 'gm_result')}
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(ser, f, indent=2, default=str)

    print(f"\n📁 Outputs: {save_dir}/")
    print("   summary.png, pipeline_*.png, results.json")
    print(f"\n{'='*70}\n DONE!\n{'='*70}")


if __name__ == "__main__":
    main()