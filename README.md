# LLM Layer Partitioning with Reinforcement Learning (v8)

Optimally partition transformer layers across heterogeneous edge devices to minimize **TPOT (Time to Process One Token)** using progressively more advanced RL agents.

## Problem

Given N transformer layers and M heterogeneous devices with different compute power and inter-device bandwidth, find a **contiguous partition** that minimizes:

```
TPOT = sum(device_compute_times) + sum(boundary_transfer_times)
```

The problem decomposes into two sub-problems:
1. **Device ordering** (permutation) — a combinatorial problem tackled by RL
2. **Layer allocation** for a given ordering — solved exactly by Dynamic Programming

## Agent Evolution (V1 → V7)

| Version | Agent | Architecture | Key Idea |
|---------|-------|-------------|----------|
| V1 | DQN | MLP Q-Network | Baseline, per-device binary selection |
| V2 | PPO-Binary | MLP Actor-Critic | PPO with binary device selection |
| V3 | PPO-Order | MLP + Pointer | One-shot ordering via pointer attention |
| V4 | PPO-AutoReg | MLP + AutoReg | Autoregressive device selection with GAE |
| V5 | Maskable PPO | MLP + Action Mask | Masked invalid actions, autoregressive |
| V6 | GNN-PPO | Edge-Conditioned GCN + Pointer | Graph topology awareness, one-shot ordering |
| V7 | GNN-AR-PPO | ECGC + Dual Agg + AutoReg | Re-runs GNN each step with dynamic features, STOP action |

### V6 — GNN-Based PPO
- Devices form a complete graph; bandwidths are edge features
- Single forward pass produces attention scores → greedy device ordering
- Best-of-N inference: sample multiple orderings, pick lowest TPOT

### V7 — Autoregressive GNN-PPO
- GNN re-runs at each selection step with updated dynamic features
- Dynamic node features: `is_selected`, `pipeline_position`, `bw_to_pipeline_tail`
- STOP action for early termination (skip suboptimal devices)
- Reward: relative to DP-sorted baseline `(dp_tpot - agent_tpot) / dp_tpot`

## Baselines

| Method | Description |
|--------|-------------|
| DP-Sorted | Sort devices by compute power, then DP-optimal partition |
| DP-Raw | Same DP but with raw device order (0,1,2,...) |
| Beam Search | Beam search over device orderings + DP |
| Brute Force | Exhaustive permutation search + DP (small device counts only) |
| Greedy | Incremental greedy layer assignment |

## Project Structure

```
agents/
  shared.py            # Constants, observation builders, reward functions
  dqn_v1.py            # V1: DQN agent
  ppo_v2.py            # V2: PPO binary selection
  ppo_v3.py            # V3: PPO one-shot ordering
  ppo_v4.py            # V4: PPO autoregressive
  maskable_ppo_v5.py   # V5: Maskable PPO autoregressive
  gnn_ppo_v6.py        # V6: GNN-based PPO
  gnn_ar_ppo_v7.py     # V7: Autoregressive GNN-PPO
baselines.py           # DP, Beam Search, Greedy, Brute Force
config.py              # All configuration (Env, Train, Eval)
environment.py         # DeviceCluster, LayerModel, TPOT computation
evaluation/
  runner.py            # Evaluation loop and metrics
  plots.py             # Training curves, comparison charts
scaling_sweep.py       # Scaling sweep across layer/device counts
training/
  base_trainer.py      # Shared training infrastructure
  train_v1~v7.py       # Per-version training loops
  train_all.py         # Orchestrator
main.py                # Entry point
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all versions and evaluate
python main.py

# Train only V6 and V7
python main.py --version v6 v7

# Quick run (fewer episodes, smaller test set)
python main.py --quick

# Evaluate saved models without retraining
python main.py --eval-only

# Scaling sweep (layers 24-64, devices 2-10)
python scaling_sweep.py
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_layers` | 32 | Transformer layers |
| `num_devices` | 5 | Edge devices |
| `device_compute_range` | (0.1, 1.0) | Device compute power range |
| `bandwidth_range` | (0.005, 0.5) | Inter-device bandwidth range (GB/s) |
| `tensor_size` | 1.0 | Activation tensor size (GB) |
| `max_training_minutes` | 15.0 | Time budget per version |
| `num_episodes` | 10000 | Training episodes per version |

## Evaluation Metrics

- **Gap to DP (%)**: `(method_tpot - dp_tpot) / dp_tpot * 100` — lower is better
- **Gap to Beam (%)**: Same formula vs beam search baseline
- **Win Rate**: Percentage of test cases where method TPOT ≤ baseline × 1.02

Test grid: layers × devices = [16, 32, 48, 64] × [3, 5, 7, 10] with 3 seeds each.

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- NumPy >= 1.24
- Matplotlib >= 3.7
- SciPy >= 1.10
