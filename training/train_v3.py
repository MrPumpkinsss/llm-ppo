"""V3 Trainer: PPO-Clip one-shot ordering + min-max DP."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.base_trainer import BaseTrainer
from agents.ppo_v3 import (
    PPOv3Network, ppo_v3_generate_ordering, ppo_v3_inference
)
from agents.shared import (
    MAX_DEVICES, MAX_LAYERS,
    build_observation, build_device_features,
)
from baselines import dp_partition, min_max_bottleneck_dp
from environment import compute_simple_tpot


class PPOv3Trainer(BaseTrainer):
    """Trainer for V3: PPO-Clip one-shot ordering + min-max DP."""

    def __init__(self, config):
        super().__init__(config, "V3-PPO-Order", max_minutes=config.max_training_minutes)
        self.cfg = config

    def _build_network(self):
        return PPOv3Network(hidden_dim=self.cfg.v3_hidden_dim).to(self.device)

    def _inference_one(self, network, devices, layers, ts, nl, nd):
        partition, tpot = ppo_v3_inference(
            network, devices, layers, ts, nl, nd, self.device, num_candidates=5
        )
        return partition, tpot

    def train(self):
        self.start_time = time.time()
        network = self._build_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.cfg.v3_learning_rate)

        total_eps = self.cfg.v3_num_episodes
        batch_size = self.cfg.v3_batch_size
        epochs = self.cfg.v3_epochs_per_update
        clip_eps = self.cfg.v3_clip_eps
        ent_coef = self.cfg.v3_entropy_coef
        val_coef = self.cfg.v3_value_coef

        network.train()
        print(f"  [V3-PPO-Order] Starting training: {total_eps} episodes")

        # Accumulate batch
        batch_obs = []
        batch_dev_feats = []
        batch_orderings = []
        batch_log_probs = []
        batch_rewards = []
        batch_values = []
        batch_nd = []

        for ep in range(1, total_eps + 1):
            if self._check_time_budget():
                print(f"  [V3-PPO-Order] Time budget reached at episode {ep}")
                break

            devices, layers, ts = self._create_random_env(seed=ep)
            nl, nd = layers.num_layers, devices.num_devices

            obs = build_observation(devices, layers, ts, nl, nd)
            dev_feats = build_device_features(devices, nd)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            dev_tensor = torch.FloatTensor(dev_feats).unsqueeze(0).to(self.device)

            orderings, log_prob, attn_scores, value = ppo_v3_generate_ordering(
                network, obs_tensor, dev_tensor, nd, deterministic=False
            )

            ordering = orderings[0]
            partition = min_max_bottleneck_dp(nl, ordering, devices, layers, ts)
            tpot = compute_simple_tpot(partition, devices, layers, ts)

            # Relative reward: positive = better than DP-sorted baseline
            dp_part = dp_partition(nl, nd, devices, layers, ts)
            dp_tpot = compute_simple_tpot(dp_part, devices, layers, ts)
            reward = (dp_tpot - tpot) / (dp_tpot + 1e-8)

            batch_obs.append(obs)
            batch_dev_feats.append(dev_feats)
            batch_orderings.append(ordering)
            batch_log_probs.append(log_prob[0].item())
            batch_rewards.append(reward)
            batch_values.append(value[0, 0].item())
            batch_nd.append(nd)

            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_tpot.append(tpot)

            # PPO update when batch is full
            if len(batch_obs) >= batch_size:
                self._ppo_update(
                    network, optimizer,
                    batch_obs, batch_dev_feats, batch_orderings,
                    batch_log_probs, batch_rewards, batch_values, batch_nd,
                    clip_eps, ent_coef, val_coef, epochs
                )
                batch_obs.clear()
                batch_dev_feats.clear()
                batch_orderings.clear()
                batch_log_probs.clear()
                batch_rewards.clear()
                batch_values.clear()
                batch_nd.clear()

            if ep % self.cfg.eval_interval == 0:
                self._evaluate(network, ep)
                self._log_progress(ep, total_eps)

            self.metrics.wall_time.append(time.time() - self.start_time)

        self._evaluate(network, total_eps)
        self._log_progress(total_eps, total_eps, "(final)")

        self._restore_best(network)
        return network, self.metrics

    def _ppo_update(self, network, optimizer, obs_list, dev_feats_list,
                    orderings_list, old_log_probs, rewards, values, nd_list,
                    clip_eps, ent_coef, val_coef, epochs):
        """PPO-Clip update for one-shot ordering."""
        obs_t = torch.FloatTensor(np.array(obs_list)).to(self.device)
        dev_t = torch.FloatTensor(np.array(dev_feats_list)).to(self.device)
        old_lp = torch.FloatTensor(old_log_probs).to(self.device)
        ret = torch.FloatTensor(rewards).to(self.device)
        old_val = torch.FloatTensor(values).to(self.device)

        advantages = ret - old_val
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            attn_scores, value_pred = network(obs_t, dev_t)

            # Recompute log probs for the stored orderings with per-sample masks
            new_log_probs = []
            total_entropy = 0.0

            for b in range(len(orderings_list)):
                ordering = orderings_list[b]
                nd = nd_list[b]
                scores = attn_scores[b]
                mask = torch.zeros(MAX_DEVICES, device=self.device, dtype=torch.bool)
                mask[:nd] = True

                lp = 0.0
                ent = 0.0
                for step, action in enumerate(ordering):
                    masked_scores = scores.clone()
                    masked_scores[~mask] = float('-inf')
                    probs = F.softmax(masked_scores.unsqueeze(0), dim=-1).squeeze(0)
                    lp += torch.log(probs[action].clamp(min=1e-8))

                    # Entropy over valid actions
                    valid_probs = probs[mask]
                    ent += -(valid_probs * torch.log(valid_probs.clamp(min=1e-8))).sum()

                    mask[action] = False

                new_log_probs.append(lp)
                total_entropy += ent / max(len(ordering), 1)

            new_lp_t = torch.stack(new_log_probs)
            avg_entropy = (total_entropy / len(orderings_list)).item()

            # PPO clip
            ratio = torch.exp(new_lp_t - old_lp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (ret - value_pred.squeeze()).pow(2).mean()
            entropy_tensor = torch.tensor(avg_entropy, device=self.device)

            loss = policy_loss + val_coef * value_loss - ent_coef * entropy_tensor

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()

            self.metrics.policy_losses.append(policy_loss.item())
            self.metrics.value_losses.append(value_loss.item())
            self.metrics.entropies.append(avg_entropy)
