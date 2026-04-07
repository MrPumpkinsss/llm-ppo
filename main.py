"""Main entry point for LLM Layer Partitioning RL Training and Evaluation.

Usage:
    python main.py                  # Train both PPO variants and evaluate
    python main.py --eval-only      # Skip training, evaluate saved models
    python main.py --quick          # Quick run with fewer episodes for testing
"""
import argparse
import os
import sys
import time
import json
import torch
import numpy as np

from config import TrainConfig, EvalConfig
from train import train_all, PPOv1Trainer, PPOv2Trainer
from evaluate import run_evaluation, generate_all_plots


def main():
    parser = argparse.ArgumentParser(description='LLM Layer Partitioning with RL')
    parser.add_argument('--eval-only', action='store_true', help='Skip training, evaluate saved models')
    parser.add_argument('--quick', action='store_true', help='Quick run for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = TrainConfig(seed=args.seed)
    eval_config = EvalConfig(output_dir=args.output_dir)

    if args.quick:
        config.v1_num_episodes = 500
        config.v2_num_episodes = 500
        config.eval_interval = 100
        config.num_eval_configs = 20
        eval_config.num_layers_to_test = [16, 32]
        eval_config.num_devices_to_test = [3, 5]

    device = config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = 'cpu'
        device = 'cpu'

    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    if not args.eval_only:
        # ====== TRAINING ======
        total_start = time.time()

        results_dict = train_all(config)

        network_v1, metrics_v1 = results_dict['ppo_v1']
        network_v2, metrics_v2 = results_dict['ppo_v2']

        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*60}")

        # Save models
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(network_v1.state_dict(), os.path.join(args.output_dir, 'ppo_v1_model.pt'))
        torch.save(network_v2.state_dict(), os.path.join(args.output_dir, 'ppo_v2_model.pt'))
        print(f"Models saved to {args.output_dir}")
    else:
        # Load models
        network_v1_file = os.path.join(args.output_dir, 'ppo_v1_model.pt')
        network_v2_file = os.path.join(args.output_dir, 'ppo_v2_model.pt')

        if not os.path.exists(network_v1_file) or not os.path.exists(network_v2_file):
            print("Error: Model files not found. Run training first.")
            sys.exit(1)

        from ppo_v1 import OrderPredictor, get_obs_dim
        from ppo_v2 import DeviceOrderNetwork

        max_devices = 10
        max_obs_dim = get_obs_dim(max_devices, 64)

        network_v1 = OrderPredictor(
            obs_dim=max_obs_dim, max_devices=max_devices,
            hidden_dim=config.v1_hidden_dim,
        ).to(device)
        network_v1.load_state_dict(torch.load(network_v1_file, map_location=device))

        network_v2 = DeviceOrderNetwork(
            max_devices=max_devices, obs_dim=max_obs_dim,
            hidden_dim=config.v2_hidden_dim, num_layers_net=config.v2_num_layers,
        ).to(device)
        network_v2.load_state_dict(torch.load(network_v2_file, map_location=device))

        metrics_v1 = metrics_v2 = None

    # ====== EVALUATION ======
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results, stats = run_evaluation(network_v1, network_v2, config, eval_config)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total test cases: {len(results)}")
    print(f"PPO-v1 ≤ DP: {stats['ppo_v1_wins']}/{len(results)} "
          f"({stats['ppo_v1_wins']/len(results)*100:.1f}%)")
    print(f"PPO-v2 ≤ DP: {stats['ppo_v2_wins']}/{len(results)} "
          f"({stats['ppo_v2_wins']/len(results)*100:.1f}%)")
    print(f"Avg gap to DP - PPO-v1: {np.mean(stats['ppo_v1_avg_gap_to_dp']):.2f}%")
    print(f"Avg gap to DP - PPO-v2: {np.mean(stats['ppo_v2_avg_gap_to_dp']):.2f}%")
    print(f"Avg gap to DP - Greedy: {np.mean(stats['ppo_v1_avg_gap_to_greedy']):.2f}%")

    # Save stats
    stats_save = {k: v for k, v in stats.items() if not isinstance(v, dict)}
    stats_save['by_num_layers'] = {str(k): {kk: np.mean(vv).tolist() for kk, vv in v.items()}
                                    for k, v in stats['by_num_layers'].items()}
    stats_save['by_num_devices'] = {str(k): {kk: np.mean(vv).tolist() for kk, vv in v.items()}
                                     for k, v in stats['by_num_devices'].items()}

    with open(os.path.join(args.output_dir, 'eval_stats.json'), 'w') as f:
        json.dump(stats_save, f, indent=2)

    # Generate plots
    generate_all_plots(results, stats, metrics_v1, metrics_v2, args.output_dir, config)

    # Final verdict
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    ppo_v1_pass = stats['ppo_v1_wins'] / len(results) >= 0.8
    ppo_v2_pass = stats['ppo_v2_wins'] / len(results) >= 0.8
    print(f"PPO-v1 meets target (≥80% ≤ DP): {'PASS' if ppo_v1_pass else 'NEED IMPROVEMENT'}")
    print(f"PPO-v2 meets target (≥80% ≤ DP): {'PASS' if ppo_v2_pass else 'NEED IMPROVEMENT'}")

    return ppo_v1_pass and ppo_v2_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
