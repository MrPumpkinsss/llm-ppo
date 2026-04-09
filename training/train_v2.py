"""V2 Trainer: PPO binary device selection + min-max DP."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.base_trainer import BaseTrainer
from agents.ppo_v2 import (
    PPOv2Network, ppo_v2_sample_action, selection_to_partition, ppo_v2_inference
)
from agents.shared import compute_reward, get_obs_dim, MAX_DEVICES, build_observation
from environment import compute_simple_tpot


class PPOv2Trainer(BaseTrainer):
    """Trainer for V2: PPO binary device selection + min-max DP."""

    def __init__(self, config):
        super().__init__(config, "V2-PPO-Binary", max_minutes=config.max_training_minutes)
        self.cfg = config

    def _build_network(self):
        return PPOv2Network(hidden_dim=self.cfg.v2_hidden_dim).to(self.device)

    def _inference_one(self, network, devices, layers, ts, nl, nd):
        partition, tpot = ppo_v2_inference(
            network, devices, layers, ts, nl, nd, self.device, num_candidates=5
        )
        return partition, tpot

    def train(self):
        self.start_time = time.time()
        network = self._build_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.cfg.v2_learning_rate)

        total_eps = self.cfg.v2_num_episodes
        batch_size = self.cfg.v2_batch_size
        epochs = self.cfg.v2_epochs_per_update
        clip_eps = self.cfg.v2_clip_eps
        ent_coef = self.cfg.v2_entropy_coef
        val_coef = self.cfg.v2_value_coef

        network.train()
        print(f"  [V2-PPO-Binary] Starting training: {total_eps} episodes")

        # Accumulate batch
        batch_obs = []
        batch_selections = []
        batch_log_probs = []
        batch_rewards = []
        batch_values = []

        for ep in range(1, total_eps + 1):
            if self._check_time_budget():
                print(f"  [V2-PPO-Binary] Time budget reached at episode {ep}")
                break

            devices, layers, ts = self._create_random_env(seed=ep)
            nl, nd = layers.num_layers, devices.num_devices

            obs = build_observation(devices, layers, ts, nl, nd)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            selection, log_prob, logits, value = ppo_v2_sample_action(
                network, obs_tensor, nd
            )

            sel_np = selection.detach().cpu().numpy().flatten()
            partition = selection_to_partition(sel_np, nl, nd, devices, layers, ts)
            tpot = compute_simple_tpot(partition, devices, layers, ts)

            reward = compute_reward(partition, devices, layers, ts)

            batch_obs.append(obs)
            batch_selections.append(sel_np)
            batch_log_probs.append(log_prob.item())
            batch_rewards.append(reward)
            batch_values.append(value.item())

            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_tpot.append(tpot)

            # PPO update when batch is full
            if len(batch_obs) >= batch_size:
                self._ppo_update(
                    network, optimizer,
                    batch_obs, batch_selections, batch_log_probs,
                    batch_rewards, batch_values,
                    clip_eps, ent_coef, val_coef, epochs, nd
                )
                batch_obs.clear()
                batch_selections.clear()
                batch_log_probs.clear()
                batch_rewards.clear()
                batch_values.clear()

            # Eval
            if ep % self.cfg.eval_interval == 0:
                self._evaluate(network, ep)
                self._log_progress(ep, total_eps)

            self.metrics.wall_time.append(time.time() - self.start_time)

        # Final eval
        self._evaluate(network, total_eps)
        self._log_progress(total_eps, total_eps, "(final)")

        self._restore_best(network)
        return network, self.metrics

    def _ppo_update(self, network, optimizer, obs, selections, old_log_probs,
                    rewards, values, clip_eps, ent_coef, val_coef, epochs, num_devices):
        """PPO-Clip update for binary selection."""
        obs_t = torch.FloatTensor(np.array(obs)).to(self.device)
        sel_t = torch.FloatTensor(np.array(selections)).to(self.device)
        old_lp = torch.FloatTensor(old_log_probs).to(self.device)
        ret = torch.FloatTensor(rewards).to(self.device)
        old_val = torch.FloatTensor(values).to(self.device)

        # Advantages
        advantages = ret - old_val
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            device_logits, value_pred = network(obs_t)
            probs = torch.sigmoid(device_logits)

            # Mask
            mask = torch.zeros_like(probs)
            mask[:, :num_devices] = 1.0
            probs_masked = probs * mask

            # Log prob of the selections
            selected_probs = probs_masked * sel_t + (1 - probs_masked) * (1 - sel_t)
            new_log_probs = torch.log(selected_probs[:, :num_devices].clamp(min=1e-8)).sum(dim=1)

            # Entropy
            p = probs_masked[:, :num_devices].clamp(min=1e-8, max=1 - 1e-8)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()

            # PPO clip
            ratio = torch.exp(new_log_probs - old_lp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (ret - value_pred.squeeze()).pow(2).mean()

            loss = policy_loss + val_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()

            self.metrics.policy_losses.append(policy_loss.item())
            self.metrics.value_losses.append(value_loss.item())
            self.metrics.entropies.append(entropy.item())
