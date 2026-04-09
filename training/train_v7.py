"""V7 Trainer: Autoregressive GNN-PPO with Positional Encoding + sum-based TPOT DP."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.base_trainer import BaseTrainer
from agents.gnn_ar_ppo_v7 import (
    PPOv7Network, ppo_v7_generate_episode, ppo_v7_inference,
    build_v7_graph_observation, STOP_ACTION,
)
from agents.shared import compute_reward, MAX_DEVICES
from baselines import min_sum_tpot_dp
from environment import compute_simple_tpot


class PPOv7Trainer(BaseTrainer):
    """Trainer for V7: Autoregressive GNN-PPO + positional encoding + sum-based TPOT DP."""

    def __init__(self, config):
        super().__init__(config, "V7-GNN-AR-PPO", max_minutes=config.max_training_minutes)
        self.cfg = config

    def _build_network(self):
        return PPOv7Network(
            hidden_dim=self.cfg.v7_hidden_dim,
            num_gnn_layers=self.cfg.v7_num_gnn_layers,
        ).to(self.device)

    def _inference_one(self, network, devices, layers, ts, nl, nd):
        partition, tpot = ppo_v7_inference(
            network, devices, layers, ts, nl, nd, self.device, num_candidates=5
        )
        return partition, tpot

    def train(self):
        self.start_time = time.time()
        network = self._build_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.cfg.v7_learning_rate)

        total_eps = self.cfg.v7_num_episodes
        batch_size = self.cfg.v7_batch_size
        epochs = self.cfg.v7_epochs_per_update
        clip_eps = self.cfg.v7_clip_eps
        gamma = self.cfg.v7_gamma
        gae_lambda = self.cfg.v7_gae_lambda
        ent_coef = self.cfg.v7_entropy_coef
        val_coef = self.cfg.v7_value_coef

        network.train()
        print(f"  [V7-GNN-AR-PPO] Starting training: {total_eps} episodes")

        # Per-step accumulators (flattened across episodes)
        all_node_feats = []
        all_edge_feats = []
        all_adj_mask = []
        all_layer_costs = []
        all_global_dynamic = []
        all_actions = []
        all_old_log_probs = []
        all_old_values = []
        all_action_masks = []
        all_returns = []
        all_advantages = []

        for ep in range(1, total_eps + 1):
            if self._check_time_budget():
                print(f"  [V7-GNN-AR-PPO] Time budget reached at episode {ep}")
                break

            devices, layers, ts = self._create_random_env(seed=ep)
            nl, nd = layers.num_layers, devices.num_devices

            # Generate episode
            step_data, ordering, partition, tpot = ppo_v7_generate_episode(
                network, devices, layers, ts, nl, nd, self.device
            )

            # Relative reward: positive = better than DP-sorted baseline
            ordered_devices = sorted(range(nd), key=lambda d: devices.compute_power[d], reverse=True)
            dp_part = min_sum_tpot_dp(nl, ordered_devices, devices, layers, ts)
            dp_tpot = compute_simple_tpot(dp_part, devices, layers, ts)
            reward = (dp_tpot - tpot) / (dp_tpot + 1e-8)

            # Compute GAE with sparse reward (reward only at last step)
            ep_length = len(step_data)
            # step_data tuple: (node_feats, edge_feats, adj_mask, layer_costs,
            #                   global_dynamic, action, log_prob, value, action_mask)
            step_values = [s[7] for s in step_data]
            step_values_aug = step_values + [0.0]

            ep_advantages = []
            gae = 0.0
            for t in reversed(range(ep_length)):
                r = reward if t == ep_length - 1 else 0.0
                delta = r + gamma * step_values_aug[t + 1] - step_values_aug[t]
                gae = delta + gamma * gae_lambda * gae
                ep_advantages.insert(0, gae)

            ep_returns = [a + v for a, v in zip(ep_advantages, step_values)]

            # Store step data
            for i, step in enumerate(step_data):
                all_node_feats.append(step[0])        # (MAX_DEVICES, 11)
                all_edge_feats.append(step[1])         # (MAX_DEVICES, MAX_DEVICES, 3)
                all_adj_mask.append(step[2])           # (MAX_DEVICES, MAX_DEVICES)
                all_layer_costs.append(step[3])        # (MAX_LAYERS,)
                all_global_dynamic.append(step[4])     # (2,)
                all_actions.append(step[5])            # int
                all_old_log_probs.append(step[6])      # float
                all_old_values.append(step[7])         # float
                all_action_masks.append(step[8])       # (num_actions,)
                all_returns.append(ep_returns[i])
                all_advantages.append(ep_advantages[i])

            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_tpot.append(tpot)

            # PPO update when enough steps accumulated
            total_steps = len(all_node_feats)
            if total_steps >= batch_size:
                self._ppo_update(
                    network, optimizer,
                    all_node_feats, all_edge_feats, all_adj_mask, all_layer_costs,
                    all_global_dynamic,
                    all_actions, all_old_log_probs, all_old_values,
                    all_action_masks, all_returns, all_advantages,
                    clip_eps, ent_coef, val_coef, epochs
                )
                all_node_feats.clear()
                all_edge_feats.clear()
                all_adj_mask.clear()
                all_layer_costs.clear()
                all_global_dynamic.clear()
                all_actions.clear()
                all_old_log_probs.clear()
                all_old_values.clear()
                all_action_masks.clear()
                all_returns.clear()
                all_advantages.clear()

            if ep % self.cfg.eval_interval == 0:
                self._evaluate(network, ep)
                self._log_progress(ep, total_eps)

            self.metrics.wall_time.append(time.time() - self.start_time)

        self._evaluate(network, total_eps)
        self._log_progress(total_eps, total_eps, "(final)")

        self._restore_best(network)
        return network, self.metrics

    def _ppo_update(
        self, network, optimizer,
        node_feats_list, edge_feats_list, adj_mask_list, layer_costs_list,
        global_dynamic_list,
        actions_list, old_log_probs_list, old_values_list,
        action_masks_list, returns_list, advantages_list,
        clip_eps, ent_coef, val_coef, epochs
    ):
        """PPO-Clip update for autoregressive GNN-PPO."""
        # Convert to batched tensors
        node_t = torch.FloatTensor(np.array(node_feats_list)).to(self.device)
        edge_t = torch.FloatTensor(np.array(edge_feats_list)).to(self.device)
        adj_t = torch.BoolTensor(np.array(adj_mask_list)).to(self.device)
        layer_t = torch.FloatTensor(np.array(layer_costs_list)).to(self.device)
        gd_t = torch.FloatTensor(np.array(global_dynamic_list)).to(self.device)
        actions_t = torch.LongTensor(actions_list).to(self.device)
        old_lp_t = torch.FloatTensor(old_log_probs_list).to(self.device)
        old_val_t = torch.FloatTensor(old_values_list).to(self.device)
        masks_t = torch.BoolTensor(np.array(action_masks_list)).to(self.device)
        ret_t = torch.FloatTensor(returns_list).to(self.device)
        adv_t = torch.FloatTensor(advantages_list).to(self.device)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(epochs):
            # Forward pass: replay each step with stored observations
            all_logits, value_pred, _ = network.forward_step(
                node_t, edge_t, adj_t, layer_t, gd_t
            )

            # Apply stored action masks
            masked_logits = all_logits.clone()
            masked_logits[~masks_t] = float('-inf')

            probs = F.softmax(masked_logits, dim=-1)

            # New log probs for taken actions
            new_log_probs = torch.log(
                probs.gather(1, actions_t.unsqueeze(1)).clamp(min=1e-8)
            ).squeeze(1)

            # Entropy over valid actions only
            valid_probs = probs.clone()
            valid_probs[~masks_t] = 0.0
            log_valid = torch.log(valid_probs.clamp(min=1e-8))
            entropy = -(valid_probs * log_valid).sum(dim=-1)
            # Normalize by number of valid actions
            num_valid = masks_t.float().sum(dim=-1).clamp(min=1)
            entropy = (entropy / num_valid).mean()

            # PPO clip
            ratio = torch.exp(new_log_probs - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * (ret_t - value_pred.squeeze()).pow(2).mean()

            # Total loss
            loss = policy_loss + val_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), self.cfg.v7_max_grad_norm)
            optimizer.step()

            self.metrics.policy_losses.append(policy_loss.item())
            self.metrics.value_losses.append(value_loss.item())
            self.metrics.entropies.append(entropy.item())
