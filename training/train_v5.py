"""V5 Trainer: Maskable PPO-Clip + sequential device selection + sum-based TPOT DP."""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.base_trainer import BaseTrainer, TrainingMetrics
from agents.maskable_ppo_v5 import (
    MaskablePPOv5Network, maskable_v5_generate_episode, maskable_v5_inference,
    STOP_ACTION, build_action_mask,
)
from agents.shared import (
    compute_reward, get_seq_obs_dim, MAX_DEVICES,
)
from baselines import min_sum_tpot_dp
from environment import compute_simple_tpot


class MaskablePPOv5Trainer(BaseTrainer):
    """Trainer for V5: Maskable PPO-Clip + sum-based TPOT DP."""

    def __init__(self, config):
        super().__init__(config, "V5-MaskPPO", max_minutes=config.max_training_minutes)
        self.cfg = config

    def _build_network(self):
        return MaskablePPOv5Network(hidden_dim=self.cfg.v5_hidden_dim).to(self.device)

    def _load_checkpoint(self, network):
        """Load network from checkpoint if available. Optimizer is NOT restored for stability."""
        model_path = os.path.join(self.cfg.checkpoint_dir, 'v5_model.pt')
        metrics_path = os.path.join(self.cfg.checkpoint_dir, 'v5_metrics.json')
        start_ep = 1

        if os.path.exists(model_path):
            network.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"  [V5-MaskPPO] Loaded network from {model_path}")

            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m_data = json.load(f)
                self.metrics = TrainingMetrics(
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
                # Resume from where we left off
                start_ep = len(self.metrics.episode_rewards) + 1
                print(f"  [V5-MaskPPO] Loaded metrics, resuming from episode {start_ep}")
                self.best_eval_reward = max(self.metrics.eval_rewards) if self.metrics.eval_rewards else float('-inf')

            return network, start_ep
        return network, start_ep

    def _inference_one(self, network, devices, layers, ts, nl, nd):
        partition, tpot = maskable_v5_inference(
            network, devices, layers, ts, nl, nd, self.device, num_candidates=5
        )
        return partition, tpot

    def train(self):
        self.start_time = time.time()
        network = self._build_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.cfg.v5_learning_rate)

        total_eps = self.cfg.v5_num_episodes
        batch_size = self.cfg.v5_batch_size
        epochs = self.cfg.v5_epochs_per_update
        clip_eps = self.cfg.v5_clip_eps
        gamma = self.cfg.v5_gamma
        gae_lambda = self.cfg.v5_gae_lambda
        ent_coef = self.cfg.v5_entropy_coef
        val_coef = self.cfg.v5_value_coef

        start_ep = 1
        if self.cfg.resume_from_checkpoint:
            network, start_ep = self._load_checkpoint(network)

        network.train()
        resumed = " (resumed)" if start_ep > 1 else ""
        print(f"  [V5-MaskPPO] Starting training: {total_eps} episodes{resumed}")

        all_step_obs = []
        all_step_actions = []
        all_step_old_log_probs = []
        all_step_values = []
        all_step_masks = []
        all_step_hiddens = []
        all_step_returns = []

        for ep in range(start_ep, total_eps + 1):
            if self._check_time_budget():
                print(f"  [V5-MaskPPO] Time budget reached at episode {ep}")
                break

            devices, layers, ts = self._create_random_env(seed=ep)
            nl, nd = layers.num_layers, devices.num_devices

            step_data, ordering, partition, tpot = maskable_v5_generate_episode(
                network, devices, layers, ts, nl, nd, self.device
            )

            reward = compute_reward(partition, devices, layers, ts)

            # Compute GAE with sparse reward
            ep_length = len(step_data)
            step_values = [s[3] for s in step_data]
            step_values_aug = step_values + [0.0]

            advantages = []
            gae = 0.0
            for t in reversed(range(ep_length)):
                # Reward only at the last step
                r = reward if t == ep_length - 1 else 0.0
                delta = r + gamma * step_values_aug[t + 1] - step_values_aug[t]
                gae = delta + gamma * gae_lambda * gae
                advantages.insert(0, gae)

            returns = [adv + sv for adv, sv in zip(advantages, step_values)]

            for i, (obs, action, log_prob, value, action_mask, h_state) in enumerate(step_data):
                all_step_obs.append(obs)
                all_step_actions.append(action)
                all_step_old_log_probs.append(log_prob)
                all_step_values.append(step_values[i])
                all_step_masks.append(action_mask)
                all_step_hiddens.append(h_state)
                all_step_returns.append(returns[i])

            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_tpot.append(tpot)

            total_steps = len(all_step_obs)
            if total_steps >= batch_size:
                self._ppo_update(
                    network, optimizer,
                    all_step_obs, all_step_actions, all_step_old_log_probs,
                    all_step_values, all_step_returns, all_step_masks,
                    all_step_hiddens,
                    clip_eps, ent_coef, val_coef, epochs
                )
                all_step_obs.clear()
                all_step_actions.clear()
                all_step_old_log_probs.clear()
                all_step_values.clear()
                all_step_masks.clear()
                all_step_hiddens.clear()
                all_step_returns.clear()

            # Save checkpoint periodically
            if ep % (self.cfg.eval_interval * 2) == 0:
                self._save_checkpoint(network)

            if ep % self.cfg.eval_interval == 0:
                self._evaluate(network, ep)
                self._log_progress(ep, total_eps)

            self.metrics.wall_time.append(time.time() - self.start_time)

        self._evaluate(network, total_eps)
        self._log_progress(total_eps, total_eps, "(final)")

        self._restore_best(network)
        return network, self.metrics

    def _save_checkpoint(self, network):
        """Save network and metrics for resuming later. Optimizer is NOT saved for stability."""
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        torch.save(network.state_dict(), os.path.join(self.cfg.checkpoint_dir, 'v5_model.pt'))
        m_dict = {
            'episode_rewards': self.metrics.episode_rewards,
            'episode_tpot': self.metrics.episode_tpot,
            'policy_losses': self.metrics.policy_losses,
            'value_losses': self.metrics.value_losses,
            'entropies': self.metrics.entropies,
            'eval_rewards': self.metrics.eval_rewards,
            'eval_tpot': self.metrics.eval_tpot,
            'dp_tpot': self.metrics.dp_tpot,
            'beam_tpot': self.metrics.beam_tpot,
            'wall_time': self.metrics.wall_time,
            'episodes_log': self.metrics.episodes_log,
        }
        with open(os.path.join(self.cfg.checkpoint_dir, 'v5_metrics.json'), 'w') as f:
            json.dump(m_dict, f)

    def _ppo_update(self, network, optimizer,
                    obs_list, action_list, old_log_probs, old_values,
                    returns, mask_list, hidden_list,
                    clip_eps, ent_coef, val_coef, epochs):
        """Maskable PPO-Clip update with stored action masks and hidden states."""
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
            # Use stored hidden states instead of zero init
            logits, value_pred, _ = network.forward_no_mask(obs_t, hiddens_t)

            # Apply stored masks (critical for correct importance sampling)
            masked_logits = logits.masked_fill(~masks_t, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)

            # Clamp for numerical stability
            new_log_probs = torch.log(
                probs.gather(1, actions_t.unsqueeze(1)).clamp(min=1e-8)
            ).squeeze(1)

            # Entropy only over valid actions
            valid_probs = probs.clone()
            valid_probs[~masks_t] = 0.0
            log_valid = torch.log(valid_probs.clamp(min=1e-8))
            entropy = -(valid_probs * log_valid).sum(dim=-1).mean()

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
