"""V4 Trainer: PPO-Clip autoregressive ordering + min-max DP."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.base_trainer import BaseTrainer
from agents.ppo_v4 import (
    PPOv4Network, ppo_v4_generate_episode, ppo_v4_inference
)
from agents.shared import (
    compute_reward, get_seq_obs_dim, MAX_DEVICES,
    build_sequential_observation, build_observation,
)
from baselines import min_max_bottleneck_dp
from environment import compute_simple_tpot


class PPOv4Trainer(BaseTrainer):
    """Trainer for V4: PPO-Clip autoregressive ordering + min-max DP."""

    def __init__(self, config):
        super().__init__(config, "V4-PPO-AutoReg", max_minutes=config.max_training_minutes)
        self.cfg = config

    def _build_network(self):
        return PPOv4Network(hidden_dim=self.cfg.v4_hidden_dim).to(self.device)

    def _inference_one(self, network, devices, layers, ts, nl, nd):
        partition, tpot = ppo_v4_inference(
            network, devices, layers, ts, nl, nd, self.device, num_candidates=5
        )
        return partition, tpot

    def train(self):
        self.start_time = time.time()
        network = self._build_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.cfg.v4_learning_rate)

        total_eps = self.cfg.v4_num_episodes
        batch_size = self.cfg.v4_batch_size
        epochs = self.cfg.v4_epochs_per_update
        clip_eps = self.cfg.v4_clip_eps
        gamma = self.cfg.v4_gamma
        gae_lambda = self.cfg.v4_gae_lambda
        ent_coef = self.cfg.v4_entropy_coef
        val_coef = self.cfg.v4_value_coef

        network.train()
        print(f"  [V4-PPO-AutoReg] Starting training: {total_eps} episodes")

        # Accumulate episodes
        all_step_obs = []
        all_step_actions = []
        all_step_old_log_probs = []
        all_step_values = []
        all_step_masks = []
        all_step_hiddens = []
        all_ep_rewards = []

        for ep in range(1, total_eps + 1):
            if self._check_time_budget():
                print(f"  [V4-PPO-AutoReg] Time budget reached at episode {ep}")
                break

            devices, layers, ts = self._create_random_env(seed=ep)
            nl, nd = layers.num_layers, devices.num_devices

            step_data, ordering, partition, tpot = ppo_v4_generate_episode(
                network, devices, layers, ts, nl, nd, self.device
            )

            reward = compute_reward(partition, devices, layers, ts)

            # Compute GAE
            step_values = [s[3] for s in step_data]
            ep_length = len(step_data)

            # Bootstrap value for last step
            step_values_aug = step_values + [0.0]  # terminal value = 0
            advantages = []
            gae = 0.0
            for t in reversed(range(ep_length)):
                delta = reward if t == ep_length - 1 else 0.0  # sparse reward
                delta += gamma * step_values_aug[t + 1] - step_values_aug[t]
                gae = delta + gamma * gae_lambda * gae
                advantages.insert(0, gae)

            for i, (obs, action, log_prob, value, mask, h_state) in enumerate(step_data):
                all_step_obs.append(obs)
                all_step_actions.append(action)
                all_step_old_log_probs.append(log_prob)
                all_step_values.append(step_values[i])
                all_step_masks.append(mask)
                all_step_hiddens.append(h_state)

            # Store episode-level reward (applied to all steps via GAE)
            returns = [a + step_values[i] for i, a in enumerate(advantages)]
            all_ep_rewards.extend(returns)

            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_tpot.append(tpot)

            # PPO update when enough steps accumulated
            total_steps = len(all_step_obs)
            if total_steps >= batch_size:
                self._ppo_update(
                    network, optimizer,
                    all_step_obs, all_step_actions, all_step_old_log_probs,
                    all_step_values, all_ep_rewards, all_step_masks,
                    all_step_hiddens,
                    clip_eps, ent_coef, val_coef, epochs
                )
                all_step_obs.clear()
                all_step_actions.clear()
                all_step_old_log_probs.clear()
                all_step_values.clear()
                all_step_masks.clear()
                all_step_hiddens.clear()
                all_ep_rewards.clear()

            if ep % self.cfg.eval_interval == 0:
                self._evaluate(network, ep)
                self._log_progress(ep, total_eps)

            self.metrics.wall_time.append(time.time() - self.start_time)

        self._evaluate(network, total_eps)
        self._log_progress(total_eps, total_eps, "(final)")

        self._restore_best(network)
        return network, self.metrics

    def _ppo_update(self, network, optimizer,
                    obs_list, action_list, old_log_probs, old_values,
                    returns, mask_list, hidden_list,
                    clip_eps, ent_coef, val_coef, epochs):
        """PPO-Clip update for autoregressive ordering."""
        obs_t = torch.FloatTensor(np.array(obs_list)).to(self.device)
        actions_t = torch.LongTensor(action_list).to(self.device)
        old_lp_t = torch.FloatTensor(old_log_probs).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        old_val_t = torch.FloatTensor(old_values).to(self.device)
        masks_t = torch.BoolTensor(np.array(mask_list)).to(self.device)
        hiddens_t = torch.FloatTensor(np.array(hidden_list)).to(self.device)

        advantages = ret_t - old_val_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            # Use stored hidden states (one per step) instead of zero init
            logits, value_pred, _ = network(obs_t, hiddens_t)

            # Apply masks
            masked_logits = logits.masked_fill(~masks_t, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)

            new_log_probs = torch.log(
                probs.gather(1, actions_t.unsqueeze(1)).clamp(min=1e-8)
            ).squeeze(1)

            entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1).mean()

            ratio = torch.exp(new_log_probs - old_lp_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (ret_t - value_pred.squeeze()).pow(2).mean()

            loss = policy_loss + val_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()

            self.metrics.policy_losses.append(policy_loss.item())
            self.metrics.value_losses.append(value_loss.item())
            self.metrics.entropies.append(entropy.item())
