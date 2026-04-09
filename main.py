"""Main entry point for LLM Layer Partitioning RL Training and Evaluation.

Usage:
    python main.py                  # Train all 5 versions and evaluate
    python main.py --eval-only      # Skip training, evaluate saved models
    python main.py --quick          # Quick run with fewer episodes
    python main.py --version v1 v3  # Train only specific versions

Resume training:
    Set resume_from_checkpoint=True in config.py TrainConfig, then run python main.py
"""
import argparse
import os
import sys
import time
import json
import torch
import numpy as np

from config import TrainConfig, EvalConfig
from training.train_all import train_all
from evaluation.runner import run_evaluation
from evaluation.plots import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description='LLM Layer Partitioning with RL')
    parser.add_argument('--eval-only', action='store_true', help='Skip training')
    parser.add_argument('--quick', action='store_true', help='Quick run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--version', nargs='*', default=None,
                        help='Train only specific versions (e.g., --version v1 v3)')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = TrainConfig(seed=args.seed, checkpoint_dir=args.output_dir)
    eval_config = EvalConfig(output_dir=args.output_dir)

    # Version selection
    if args.version:
        versions = [v.lower().replace('v', '') for v in args.version]
        config.train_v1 = '1' in versions
        config.train_v2 = '2' in versions
        config.train_v3 = '3' in versions
        config.train_v4 = '4' in versions
        config.train_v5 = '5' in versions
        config.train_v6 = '6' in versions
        config.train_v7 = '7' in versions

    if args.quick:
        config.v1_num_episodes = 500
        config.v2_num_episodes = 500
        config.v3_num_episodes = 500
        config.v4_num_episodes = 500
        config.v5_num_episodes = 500
        config.v6_num_episodes = 500
        config.v7_num_episodes = 500
        config.eval_interval = 100
        config.num_eval_configs = 15
        config.max_training_minutes = 2.0
        eval_config.num_layers_to_test = [16, 32]
        eval_config.num_devices_to_test = [3, 5]
        eval_config.num_test_configs = 20

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

    networks = {}

    if not args.eval_only:
        # ====== TRAINING ======
        total_start = time.time()

        results_dict = train_all(config)

        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*60}")

        # Save models and metrics
        os.makedirs(args.output_dir, exist_ok=True)

        metrics_dict = {}
        for ver, (net, metrics) in results_dict.items():
            model_path = os.path.join(args.output_dir, f'{ver}_model.pt')
            torch.save(net.state_dict(), model_path)

            metrics_dict[ver] = metrics
            # Save metrics JSON
            m_dict = {
                'episode_rewards': metrics.episode_rewards,
                'episode_tpot': metrics.episode_tpot,
                'policy_losses': metrics.policy_losses,
                'value_losses': metrics.value_losses,
                'entropies': metrics.entropies,
                'eval_rewards': metrics.eval_rewards,
                'eval_tpot': metrics.eval_tpot,
                'dp_tpot': metrics.dp_tpot,
                'beam_tpot': metrics.beam_tpot,
                'wall_time': metrics.wall_time,
                'episodes_log': metrics.episodes_log,
            }
            with open(os.path.join(args.output_dir, f'{ver}_metrics.json'), 'w') as f:
                json.dump(m_dict, f)

        print(f"Models and metrics saved to {args.output_dir}")
    else:
        # Load models
        from agents.dqn_v1 import DQNv1Network
        from agents.ppo_v2 import PPOv2Network
        from agents.ppo_v3 import PPOv3Network
        from agents.ppo_v4 import PPOv4Network
        from agents.maskable_ppo_v5 import MaskablePPOv5Network
        from agents.gnn_ppo_v6 import PPOv6Network
        from agents.gnn_ar_ppo_v7 import PPOv7Network
        from training.base_trainer import TrainingMetrics

        torch_device = torch.device(device)

        for ver, NetClass in [('v1', DQNv1Network), ('v2', PPOv2Network),
                               ('v3', PPOv3Network), ('v4', PPOv4Network),
                               ('v5', MaskablePPOv5Network), ('v6', PPOv6Network),
                               ('v7', PPOv7Network)]:
            model_path = os.path.join(args.output_dir, f'{ver}_model.pt')
            if os.path.exists(model_path):
                net = NetClass().to(torch_device)
                net.load_state_dict(torch.load(model_path, map_location=torch_device))
                net.eval()
                networks[ver] = (net, None)
            else:
                print(f"Warning: {model_path} not found, skipping {ver}")

        # Load metrics
        metrics_dict = {}
        for ver in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']:
            mfile = os.path.join(args.output_dir, f'{ver}_metrics.json')
            if os.path.exists(mfile):
                with open(mfile) as f:
                    m_data = json.load(f)
                m = TrainingMetrics(
                    episode_rewards=m_data.get('episode_rewards', []),
                    episode_tpot=m_data.get('episode_tpot', []),
                    policy_losses=m_data.get('policy_losses', []),
                    value_losses=m_data.get('value_losses', []),
                    entropies=m_data.get('entropies', []),
                    eval_rewards=m_data.get('eval_rewards', []),
                    eval_tpot=m_data.get('eval_tpot', []),
                    dp_tpot=m_data.get('dp_tpot', []),
                    beam_tpot=m_data.get('beam_tpot', []),
                    wall_time=m_data.get('wall_time', []),
                    episodes_log=m_data.get('episodes_log', []),
                )
                metrics_dict[ver] = m

        # Reconstruct results_dict from loaded networks
        results_dict = networks

    # ====== EVALUATION ======
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results, stats = run_evaluation(results_dict, config, eval_config)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total test cases: {stats.get('total_tests', 0)}")
    print()

    for ver in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']:
        name = {'v1': 'V1-DQN', 'v2': 'V2-PPO-Binary', 'v3': 'V3-PPO-Order',
                'v4': 'V4-PPO-AutoReg', 'v5': 'V5-MaskPPO', 'v6': 'V6-GNN-PPO',
                'v7': 'V7-GNN-AR-PPO'}[ver]
        wr_dp = stats.get(f'{ver}_win_rate_dp', 0)
        wr_beam = stats.get(f'{ver}_win_rate_beam', 0)
        gap = stats.get(f'{ver}_avg_gap_to_beam', float('inf'))
        avg_t = stats.get(f'{ver}_avg_time_ms', 0)
        print(f"  {name}: Win vs DP={wr_dp:.0f}% | Win vs Beam={wr_beam:.0f}% | "
              f"Avg gap to Beam={gap:.1f}% | Avg time={avg_t:.1f}ms")

    # Baseline comparisons
    dp_raw_gap = stats.get('dp_raw_avg_gap_to_beam', float('inf'))
    dp_raw_tpot = stats.get('dp_raw_avg_tpot', 0)
    dp_tpot = stats.get('dp_avg_tpot', 0)
    beam_tpot = stats.get('beam_avg_tpot', 0)
    print(f"\n  Baselines:")
    print(f"    DP-Sorted: avg TPOT={dp_tpot:.3f}")
    print(f"    DP-Raw:    avg TPOT={dp_raw_tpot:.3f}, gap to Beam={dp_raw_gap:.1f}%")
    print(f"    BeamSearch: avg TPOT={beam_tpot:.3f}")

    # Save stats
    stats_save = {}
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            stats_save[k] = v
        elif isinstance(v, list):
            stats_save[k] = v
        elif isinstance(v, dict):
            stats_save[k] = {str(kk): vv for kk, vv in v.items()}

    with open(os.path.join(args.output_dir, 'eval_stats.json'), 'w') as f:
        json.dump(stats_save, f, indent=2, default=str)

    # Generate plots
    generate_all_plots(results, stats, metrics_dict, args.output_dir, config)

    # Final verdict
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    any_pass = False
    for ver in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']:
        wr = stats.get(f'{ver}_win_rate_beam', 0)
        passed = wr >= 50
        any_pass = any_pass or passed
        name = {'v1': 'V1-DQN', 'v2': 'V2-PPO-Binary', 'v3': 'V3-PPO-Order',
                'v4': 'V4-PPO-AutoReg', 'v5': 'V5-MaskPPO', 'v6': 'V6-GNN-PPO'}[ver]
        print(f"  {name}: {'PASS' if passed else 'NEED IMPROVEMENT'} "
              f"(win vs Beam={wr:.0f}%, target≥50%)")

    return any_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
