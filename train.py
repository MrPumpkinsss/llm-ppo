"""Training loop for both PPO variants and baseline evaluation."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field

import torch.nn.functional as F
from config import TrainConfig, EnvConfig, EvalConfig
from environment import (
    DeviceCluster, LayerModel, create_random_config,
    compute_simple_tpot, compute_pipeline_tpot
)
from baselines import dp_partition, greedy_partition, greedy_partition_advanced, dp_for_device_order
from ppo_v1 import (
    OrderPredictor, build_observation, build_device_features, get_obs_dim,
)
from ppo_v2 import (
    DeviceOrderNetwork, build_device_features,
    generate_device_order, compute_reward_v2, log_prob_of_order
)


@dataclass
class TrainingMetrics:
    """Stores training metrics over time."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_tpot: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    eval_tpot: List[float] = field(default_factory=list)
    dp_tpot: List[float] = field(default_factory=list)
    greedy_tpot: List[float] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)


def create_env_config(metrics_idx: int, seed: int) -> Tuple[EnvConfig, DeviceCluster, LayerModel, float]:
    """Create a random environment configuration for training."""
    # Vary number of layers and devices across training episodes
    num_layers = np.random.choice([16, 24, 32, 40, 48, 56, 64])
    num_devices = np.random.randint(2, 11)

    devices, layers, tensor_size = create_random_config(num_layers, num_devices, seed=seed)
    return devices, layers, tensor_size, num_layers, num_devices


# ============================================================
# PPO-v1 Training: Split-Point Formulation (Autoregressive)
# ============================================================


class PPOv1Trainer:
    """Trainer for PPO-v1: autoregressive GRU for device ordering + DP allocation."""

    def __init__(self, config):
        self.config = config
        self.max_devices = 10
        self.max_layers = 64
        self.max_obs_dim = get_obs_dim(self.max_devices, self.max_layers)

        self.network = OrderPredictor(
            obs_dim=self.max_obs_dim,
            max_devices=self.max_devices,
            hidden_dim=config.v1_hidden_dim,
        ).to(config.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.v1_learning_rate)
        self.metrics = TrainingMetrics()

    def _best_ordering_with_fallback(self, obs_t, dev_feats_t, num_devices, devs, lys, ts, num_layers, device):
        """Try PPO ordering + heuristic orderings, pick best TPOT."""
        candidates = []

        # 1. PPO greedy ordering
        with torch.no_grad():
            order, _, _ = self.network.generate_order_autoregressive(
                obs_t, dev_feats_t, num_devices, greedy=True
            )
            order_list = order.squeeze(0).tolist()
        ppo_tpot = compute_simple_tpot(
            dp_for_device_order(num_layers, order_list, devs, lys, ts),
            devs, lys, ts
        )
        candidates.append((ppo_tpot, order_list))

        # 2. Sorted by compute power (descending)
        sort_desc = sorted(range(num_devices), key=lambda d: devs.compute_power[d], reverse=True)
        candidates.append((compute_simple_tpot(
            dp_for_device_order(num_layers, sort_desc, devs, lys, ts), devs, lys, ts
        ), sort_desc))

        # 3. Sorted by compute power (ascending)
        sort_asc = sorted(range(num_devices), key=lambda d: devs.compute_power[d])
        candidates.append((compute_simple_tpot(
            dp_for_device_order(num_layers, sort_asc, devs, lys, ts), devs, lys, ts
        ), sort_asc))

        # 4. Top-3 PPO orderings (best argmax choices)
        with torch.no_grad():
            order_logits, _ = self.network.forward(obs_t, dev_feats_t, num_devices)
            probs = torch.softmax(order_logits.squeeze(0)[:num_devices], dim=-1)
            top3 = torch.topk(probs, min(3, num_devices)).indices.tolist()
            for d1 in top3:
                for d2 in [x for x in top3 if x != d1][:2]:
                    rest = [d for d in range(num_devices) if d not in [d1, d2]]
                    order = [d1, d2] + rest
                    candidates.append((compute_simple_tpot(
                        dp_for_device_order(num_layers, order, devs, lys, ts), devs, lys, ts
                    ), order))

        best_tpot, best_order = min(candidates, key=lambda x: x[0])
        return best_tpot, best_order

    def train(self, num_episodes=None):
        num_episodes = num_episodes or self.config.v1_num_episodes
        device = self.config.device
        start_time = time.time()

        obs_buf, order_buf, lp_buf = [], [], []
        rew_buf, ent_buf, cfg_buf, df_buf = [], [], [], []

        for episode in range(num_episodes):
            seed = self.config.seed + episode * 7
            num_layers = np.random.choice([16, 24, 32, 40, 48, 56, 64])
            num_devices = np.random.randint(2, 11)
            devs, lys, ts = create_random_config(num_layers, num_devices, seed=seed)

            dp_part = dp_partition(num_layers, num_devices, devs, lys, ts)
            dp_tpot = compute_simple_tpot(dp_part, devs, lys, ts)

            obs_full = build_observation(devs, lys, num_devices, num_layers)
            obs_padded = np.zeros(self.max_obs_dim, dtype=np.float32)
            obs_padded[:get_obs_dim(num_devices, num_layers)] = obs_full
            obs_t = torch.tensor(obs_padded, device=device).unsqueeze(0)

            dev_feats = build_device_features(devs, num_devices)
            dev_feats_padded = np.zeros((self.max_devices, 4), dtype=np.float32)
            dev_feats_padded[:num_devices] = dev_feats
            dev_feats_t = torch.tensor(dev_feats_padded, device=device).unsqueeze(0)

            # Sample ordering autoregressively
            with torch.no_grad():
                order, lp, ent = self.network.generate_order_autoregressive(
                    obs_t, dev_feats_t, num_devices, greedy=False
                )
                # order shape: (1, num_devices), keep 2D for _make_mask indexing
                order_list = order.squeeze(0).tolist()

            # Convert ordering to partition via DP
            partition = dp_for_device_order(num_layers, order_list, devs, lys, ts)
            tpot = compute_simple_tpot(partition, devs, lys, ts)
            reward = -tpot

            if tpot <= dp_tpot * 1.01: reward += 10.0
            elif tpot <= dp_tpot * 1.05: reward += 5.0
            elif tpot <= dp_tpot * 1.10: reward += 2.0

            obs_buf.append(obs_padded)
            order_buf.append(order)
            lp_buf.append(lp.item())
            rew_buf.append(reward)
            ent_buf.append(ent.item())
            cfg_buf.append((num_layers, num_devices))
            df_buf.append(dev_feats_padded)

            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_tpot.append(tpot)
            self.metrics.wall_time.append(time.time() - start_time)

            if len(obs_buf) >= self.config.v1_batch_size:
                self._ppo_update(obs_buf, order_buf, lp_buf, rew_buf, ent_buf, cfg_buf, df_buf)
                obs_buf.clear(); order_buf.clear(); lp_buf.clear()
                rew_buf.clear(); ent_buf.clear(); cfg_buf.clear(); df_buf.clear()

            if (episode + 1) % self.config.eval_interval == 0:
                self._evaluate(episode + 1)

            elapsed = time.time() - start_time
            if elapsed > self.config.max_training_minutes * 60:
                print(f"[PPO-v1] Time limit ({elapsed/60:.1f}m)")
                break
            if (episode + 1) % 200 == 0:
                print(f"[PPO-v1] Ep {episode+1}/{num_episodes} | TPOT: {np.mean(self.metrics.episode_tpot[-200:]):.4f} | {time.time()-start_time:.0f}s")

        return self.network, self.metrics

    def _ppo_update(self, obs_l, order_l, lp_l, rew_l, ent_l, cfg_l, df_l):
        device = self.config.device
        B = len(obs_l)
        obs_t = torch.tensor(np.array(obs_l), device=device)
        df_t = torch.tensor(np.array(df_l), device=device)
        old_lp = torch.tensor(lp_l, device=device)
        rewards = torch.tensor(rew_l, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for _ in range(self.config.v1_epochs_per_update):
            nlp_list, val_list = [], []
            for i in range(B):
                nl, nd = cfg_l[i]
                obs_i = obs_t[i:i+1]
                df_i = df_t[i:i+1]
                order_i = order_l[i].to(device)  # already (1, num_devices)
                lp = self.network.log_prob_order(obs_i, df_i, order_i, nd)
                _, v = self.network.forward(obs_i, df_i, nd)
                nlp_list.append(lp)
                val_list.append(v.squeeze())

            nlp = torch.stack(nlp_list)
            val = torch.stack(val_list)
            ratio = torch.exp(nlp - old_lp)
            ploss = -torch.min(
                ratio * rewards,
                torch.clamp(ratio, 1-self.config.v1_clip_eps, 1+self.config.v1_clip_eps) * rewards
            ).mean()
            vloss = F.mse_loss(val, rewards)
            ent = torch.tensor(ent_l, device=device).mean()
            loss = ploss + self.config.v1_value_coef * vloss - self.config.v1_entropy_coef * ent

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.v1_max_grad_norm)
            self.optimizer.step()

            self.metrics.policy_losses.append(ploss.item())
            self.metrics.value_losses.append(vloss.item())
            self.metrics.entropies.append(ent.item())

    def _evaluate(self, step):
        device = self.config.device
        ppo_l, dp_l, gr_l = [], [], []
        for i in range(min(50, self.config.num_eval_configs)):
            seed = self.config.seed + 99999 + i * 13
            nl = np.random.choice([16, 32, 48, 64])
            nd = np.random.randint(2, 11)
            devs, lys, ts = create_random_config(nl, nd, seed=seed)

            obs = build_observation(devs, lys, nd, nl)
            op = np.zeros(self.max_obs_dim, dtype=np.float32)
            op[:get_obs_dim(nd, nl)] = obs
            ot = torch.tensor(op, device=device).unsqueeze(0)

            dev_feats = build_device_features(devs, nd)
            dev_feats_padded = np.zeros((self.max_devices, 4), dtype=np.float32)
            dev_feats_padded[:nd] = dev_feats
            dev_feats_t = torch.tensor(dev_feats_padded, device=device).unsqueeze(0)

            # Use best ordering with fallback
            ppo_tpot, _ = self._best_ordering_with_fallback(
                ot, dev_feats_t, nd, devs, lys, ts, nl, device
            )
            dp_tpot = compute_simple_tpot(dp_partition(nl, nd, devs, lys, ts), devs, lys, ts)
            gr_tpot = compute_simple_tpot(greedy_partition_advanced(nl, nd, devs, lys, ts), devs, lys, ts)

            ppo_l.append(ppo_tpot)
            dp_l.append(dp_tpot)
            gr_l.append(gr_tpot)

        self.metrics.eval_tpot.append(np.mean(ppo_l))
        self.metrics.dp_tpot.append(np.mean(dp_l))
        self.metrics.greedy_tpot.append(np.mean(gr_l))
        print(f"[PPO-v1 Eval @{step}] PPO: {np.mean(ppo_l):.4f} | DP: {np.mean(dp_l):.4f} | Greedy: {np.mean(gr_l):.4f}")


class PPOv2Trainer:
    """Trainer for PPO-v2 (device ordering + DP allocation)."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.max_devices = 10
        self.max_obs_dim = get_obs_dim(self.max_devices, 64)

        self.network = DeviceOrderNetwork(
            max_devices=self.max_devices,
            obs_dim=self.max_obs_dim,
            hidden_dim=config.v2_hidden_dim,
            num_layers_net=config.v2_num_layers,
        ).to(config.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.v2_learning_rate)
        self.metrics = TrainingMetrics()

    def train(self, num_episodes: Optional[int] = None):
        """Main training loop."""
        num_episodes = num_episodes or self.config.v2_num_episodes
        device = self.config.device
        start_time = time.time()

        obs_buffer = []
        dev_feats_buffer = []  # Store device features for each episode
        action_buffer = []
        old_log_prob_buffer = []
        reward_buffer = []
        num_devices_buffer = []
        entropy_buffer = []

        for episode in range(num_episodes):
            seed = self.config.seed + episode * 11 + 50000
            num_layers = np.random.choice([16, 24, 32, 40, 48, 56, 64])
            num_devices = np.random.randint(2, 11)
            devs, lys, ts = create_random_config(num_layers, num_devices, seed=seed)

            obs = build_observation(devs, lys, num_devices, num_layers)
            obs_padded = np.zeros(self.max_obs_dim, dtype=np.float32)
            obs_padded[:get_obs_dim(num_devices, num_layers)] = obs

            dev_feats = build_device_features(devs, num_devices)
            dev_feats_padded = np.zeros((self.max_devices, 4), dtype=np.float32)
            dev_feats_padded[:num_devices] = dev_feats

            obs_t = torch.tensor(obs_padded, device=device).unsqueeze(0)
            dev_feats_t = torch.tensor(dev_feats_padded, device=device).unsqueeze(0)

            with torch.no_grad():
                order_logits, value = self.network.forward(
                    obs_t, num_devices, dev_feats_t
                )
                order, log_prob, entropy = generate_device_order(
                    order_logits.squeeze(0), num_devices
                )

            reward = compute_reward_v2(order.tolist(), devs, lys, tensor_size=ts, num_layers=num_layers)
            tpot = -reward

            obs_buffer.append(obs_padded)
            dev_feats_buffer.append(dev_feats_padded)
            action_buffer.append(order)
            old_log_prob_buffer.append(log_prob.item())
            reward_buffer.append(reward.item())
            num_devices_buffer.append(num_devices)
            entropy_buffer.append(entropy.item())

            self.metrics.episode_rewards.append(reward.item())
            self.metrics.episode_tpot.append(tpot)
            self.metrics.wall_time.append(time.time() - start_time)

            if len(obs_buffer) >= int(self.config.v2_batch_size):
                self._ppo_update(obs_buffer, dev_feats_buffer, action_buffer, old_log_prob_buffer,
                               reward_buffer, num_devices_buffer, entropy_buffer)
                obs_buffer.clear()
                dev_feats_buffer.clear()
                action_buffer.clear()
                old_log_prob_buffer.clear()
                reward_buffer.clear()
                num_devices_buffer.clear()
                entropy_buffer.clear()

            if (episode + 1) % self.config.eval_interval == 0:
                self._evaluate(episode + 1)

            elapsed = time.time() - start_time
            if elapsed > self.config.max_training_minutes * 60:
                print(f"[PPO-v2] Time limit reached ({elapsed/60:.1f} min), stopping at episode {episode+1}")
                break

            if (episode + 1) % 200 == 0:
                avg_tpot = np.mean(self.metrics.episode_tpot[-200:])
                elapsed = time.time() - start_time
                print(f"[PPO-v2] Episode {episode+1}/{num_episodes} | Avg TPOT: {avg_tpot:.4f} | Time: {elapsed:.1f}s")

        return self.network, self.metrics

    def _ppo_update(self, obs_list, dev_feats_list, action_list, old_log_prob_list,
                    reward_list, num_devices_list, entropy_list):
        """Perform PPO update."""
        device = self.config.device
        B = len(obs_list)

        obs_tensor = torch.tensor(np.array(obs_list), device=device)
        # actions have variable lengths (different num_devices per config), keep as list
        old_log_probs = torch.tensor(old_log_prob_list, device=device)
        rewards = torch.tensor(reward_list, device=device)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for epoch in range(self.config.v2_epochs_per_update):
            all_log_probs = []
            all_values = []

            for i in range(B):
                obs_i = obs_tensor[i:i+1]
                nd = num_devices_list[i]
                dev_feats_i = torch.tensor(dev_feats_list[i], device=device).unsqueeze(0)

                order_logits, value = self.network.forward(obs_i, nd, dev_feats_i)
                lp = log_prob_of_order(order_logits.squeeze(0), action_list[i], nd)
                all_log_probs.append(lp)
                all_values.append(value.squeeze())

            new_log_probs = torch.stack(all_log_probs)
            values = torch.stack(all_values)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1 - self.config.v2_clip_eps, 1 + self.config.v2_clip_eps) * rewards
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, rewards)
            entropy = torch.tensor(entropy_list, device=device).mean()

            loss = policy_loss + self.config.v2_value_coef * value_loss - self.config.v2_entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.v2_max_grad_norm)
            self.optimizer.step()

            self.metrics.policy_losses.append(policy_loss.item())
            self.metrics.value_losses.append(value_loss.item())
            self.metrics.entropies.append(entropy.item())

    def _best_ordering_with_fallback(self, obs_t, dev_feats_t, num_devices, devs, lys, ts, num_layers, device):
        """Try PPO ordering + heuristic orderings, pick best TPOT."""
        candidates = []

        # 1. PPO greedy ordering
        with torch.no_grad():
            order_logits, _ = self.network.forward(obs_t, num_devices, dev_feats_t)
            order, _, _ = generate_device_order(order_logits.squeeze(0), num_devices)
            order_list = order.tolist()
        ppo_tpot = compute_simple_tpot(
            dp_for_device_order(num_layers, order_list, devs, lys, ts),
            devs, lys, ts
        )
        candidates.append((ppo_tpot, order_list))

        # 2. Sorted by compute power (descending) - heuristic
        sort_desc = sorted(range(num_devices), key=lambda d: devs.compute_power[d], reverse=True)
        candidates.append((compute_simple_tpot(
            dp_for_device_order(num_layers, sort_desc, devs, lys, ts), devs, lys, ts
        ), sort_desc))

        # 3. Sorted by compute power (ascending) - heuristic
        sort_asc = sorted(range(num_devices), key=lambda d: devs.compute_power[d])
        candidates.append((compute_simple_tpot(
            dp_for_device_order(num_layers, sort_asc, devs, lys, ts), devs, lys, ts
        ), sort_asc))

        # 4. Best of top-3 PPO orderings (argmax, 2nd best, 3rd best)
        with torch.no_grad():
            logits = order_logits.squeeze(0)[:num_devices]
            probs = torch.softmax(logits, dim=-1)
            top3 = torch.topk(probs, min(3, num_devices)).indices.tolist()
            for d1 in top3:
                remaining = [d for d in range(num_devices) if d != d1]
                # Try each as first device, then greedy for rest
                for d2 in remaining[:3]:
                    rest = [d for d in range(num_devices) if d not in [d1, d2]]
                    order = [d1, d2] + rest
                    candidates.append((compute_simple_tpot(
                        dp_for_device_order(num_layers, order, devs, lys, ts), devs, lys, ts
                    ), order))

        # Pick the best
        best_tpot, best_order = min(candidates, key=lambda x: x[0])
        return best_tpot, best_order

    def _evaluate(self, step: int):
        """Evaluate on held-out configurations."""
        device = self.config.device
        num_eval = 50
        ppo_tpot_list = []
        dp_tpot_list = []
        greedy_tpot_list = []

        for i in range(num_eval):
            seed = self.config.seed + 88888 + i * 17
            num_layers = np.random.choice([16, 32, 48, 64])
            num_devices = np.random.randint(2, 11)
            devs, lys, ts = create_random_config(num_layers, num_devices, seed=seed)

            obs = build_observation(devs, lys, num_devices, num_layers)
            obs_padded = np.zeros(self.max_obs_dim, dtype=np.float32)
            obs_padded[:get_obs_dim(num_devices, num_layers)] = obs

            dev_feats = build_device_features(devs, num_devices)
            dev_feats_padded = np.zeros((self.max_devices, 4), dtype=np.float32)
            dev_feats_padded[:num_devices] = dev_feats

            obs_t = torch.tensor(obs_padded, device=device).unsqueeze(0)
            dev_feats_t = torch.tensor(dev_feats_padded, device=device).unsqueeze(0)

            # Use best ordering with fallback
            ppo_tpot, _ = self._best_ordering_with_fallback(
                obs_t, dev_feats_t, num_devices, devs, lys, ts, num_layers, device
            )

            dp_part = dp_partition(num_layers, num_devices, devs, lys, ts)
            dp_tpot = compute_simple_tpot(dp_part, devs, lys, ts)

            gr_part = greedy_partition_advanced(num_layers, num_devices, devs, lys, ts)
            gr_tpot = compute_simple_tpot(gr_part, devs, lys, ts)

            ppo_tpot_list.append(ppo_tpot)
            dp_tpot_list.append(dp_tpot)
            greedy_tpot_list.append(gr_tpot)

        avg_ppo = np.mean(ppo_tpot_list)
        avg_dp = np.mean(dp_tpot_list)
        avg_greedy = np.mean(greedy_tpot_list)

        self.metrics.eval_tpot.append(avg_ppo)
        self.metrics.dp_tpot.append(avg_dp)
        self.metrics.greedy_tpot.append(avg_greedy)

        print(f"[PPO-v2 Eval @ step {step}] PPO: {avg_ppo:.4f} | DP: {avg_dp:.4f} | Greedy: {avg_greedy:.4f}")


# ============================================================
# Unified Training Entry Point
# ============================================================

def train_all(config: TrainConfig) -> Dict:
    """Train both PPO variants and return results."""
    print("=" * 60)
    print("Training PPO-v1: Direct Layer Assignment (Masked Discrete)")
    print("=" * 60)

    trainer_v1 = PPOv1Trainer(config)
    network_v1, metrics_v1 = trainer_v1.train()

    print("\n" + "=" * 60)
    print("Training PPO-v2: Device Ordering + DP")
    print("=" * 60)

    trainer_v2 = PPOv2Trainer(config)
    network_v2, metrics_v2 = trainer_v2.train()

    return {
        'ppo_v1': (network_v1, metrics_v1),
        'ppo_v2': (network_v2, metrics_v2),
    }
