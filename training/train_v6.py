"""V6 Trainer: GNN-Based PPO + Edge-Conditioned Graph Conv + sum-based TPOT DP."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.base_trainer import BaseTrainer
from agents.gnn_ppo_v6 import (
    PPOv6Network, ppo_v6_generate_ordering, ppo_v6_inference,
    build_graph_observation,
)
from agents.shared import compute_reward, MAX_DEVICES
from baselines import min_sum_tpot_dp
from environment import compute_simple_tpot


class PPOv6Trainer(BaseTrainer):
    """Trainer for V6: GNN-Based PPO + sum-based TPOT DP."""

    def __init__(self, config):
        super().__init__(config, "V6-GNN-PPO", max_minutes=config.max_training_minutes)
        self.cfg = config

    def _build_network(self):
        return PPOv6Network(
            hidden_dim=self.cfg.v6_hidden_dim,
            num_gnn_layers=self.cfg.v6_num_gnn_layers,
        ).to(self.device)

    def _inference_one(self, network, devices, layers, ts, nl, nd):
        partition, tpot = ppo_v6_inference(
            network, devices, layers, ts, nl, nd, self.device, num_candidates=5
        )
        return partition, tpot

    def train(self):
        self.start_time = time.time()
        network = self._build_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.cfg.v6_learning_rate)

        total_eps = self.cfg.v6_num_episodes
        batch_size = self.cfg.v6_batch_size
        epochs = self.cfg.v6_epochs_per_update
        clip_eps = self.cfg.v6_clip_eps
        ent_coef = self.cfg.v6_entropy_coef
        val_coef = self.cfg.v6_value_coef

        network.train()
        print(f"  [V6-GNN-PPO] Starting training: {total_eps} episodes")

        # Accumulate batch
        batch_node = []
        batch_edge = []
        batch_adj = []
        batch_layer = []
        batch_orderings = []
        batch_log_probs = []
        batch_rewards = []
        batch_values = []
        batch_nd = []
        dp_cache = {}  # cache dp_tpot by seed to avoid recomputation

        for ep in range(1, total_eps + 1):
            if self._check_time_budget():
                print(f"  [V6-GNN-PPO] Time budget reached at episode {ep}")
                break

            devices, layers, ts = self._create_random_env(seed=ep)
            nl, nd = layers.num_layers, devices.num_devices

            # Build graph inputs
            node_feats, edge_feats, adj_mask, layer_costs = build_graph_observation(
                devices, layers, ts, nl, nd
            )
            node_t = torch.FloatTensor(node_feats).unsqueeze(0).to(self.device)
            edge_t = torch.FloatTensor(edge_feats).unsqueeze(0).to(self.device)
            adj_t = torch.BoolTensor(adj_mask).unsqueeze(0).to(self.device)
            layer_t = torch.FloatTensor(layer_costs).unsqueeze(0).to(self.device)

            # Generate ordering
            orderings, log_prob, attn_scores, value = ppo_v6_generate_ordering(
                network, node_t, edge_t, adj_t, layer_t, nd, deterministic=False
            )

            ordering = orderings[0]
            partition = min_sum_tpot_dp(nl, ordering, devices, layers, ts)
            tpot = compute_simple_tpot(partition, devices, layers, ts)

            reward = -tpot
            # Penalize if not beating DP baseline (cached by seed)
            """
            if ep not in dp_cache:
                dp_tpot, _ = self._compute_baselines(devices, layers, ts, nl, nd)
                dp_cache[ep] = dp_tpot
            if tpot >= dp_cache[ep]:
                reward -= 1.0
            """
            batch_node.append(node_feats)
            batch_edge.append(edge_feats)
            batch_adj.append(adj_mask)
            batch_layer.append(layer_costs)
            batch_orderings.append(ordering)
            batch_log_probs.append(log_prob[0].item())
            batch_rewards.append(reward)
            batch_values.append(value[0, 0].item())
            batch_nd.append(nd)

            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_tpot.append(tpot)

            # PPO update when batch is full
            if len(batch_node) >= batch_size:
                self._ppo_update(
                    network, optimizer,
                    batch_node, batch_edge, batch_adj, batch_layer,
                    batch_orderings, batch_log_probs, batch_rewards,
                    batch_values, batch_nd,
                    clip_eps, ent_coef, val_coef, epochs
                )
                batch_node.clear()
                batch_edge.clear()
                batch_adj.clear()
                batch_layer.clear()
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

    def _ppo_update(self, network, optimizer,
                    node_list, edge_list, adj_list, layer_list,
                    orderings_list, old_log_probs, rewards, values, nd_list,
                    clip_eps, ent_coef, val_coef, epochs):
        """PPO-Clip update for GNN-based one-shot ordering."""
        node_t = torch.FloatTensor(np.array(node_list)).to(self.device)
        edge_t = torch.FloatTensor(np.array(edge_list)).to(self.device)
        adj_t = torch.BoolTensor(np.array(adj_list)).to(self.device)
        layer_t = torch.FloatTensor(np.array(layer_list)).to(self.device)
        old_lp = torch.FloatTensor(old_log_probs).to(self.device)
        ret = torch.FloatTensor(rewards).to(self.device)
        old_val = torch.FloatTensor(values).to(self.device)

        advantages = ret - old_val
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            attn_scores, value_pred = network(node_t, edge_t, adj_t, layer_t)

            # Recompute log probs for stored orderings with per-sample masks
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
            nn.utils.clip_grad_norm_(network.parameters(), self.cfg.v6_max_grad_norm)
            optimizer.step()

            self.metrics.policy_losses.append(policy_loss.item())
            self.metrics.value_losses.append(value_loss.item())
            self.metrics.entropies.append(avg_entropy)
