"""V1 Trainer: DQN + sum-based TPOT DP."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.base_trainer import BaseTrainer
from agents.dqn_v1 import (
    DQNv1Network, ReplayBuffer, dqn_v1_generate_episode, dqn_v1_inference
)
from agents.shared import compute_reward, MAX_DEVICES
from baselines import min_sum_tpot_dp, dp_partition, beam_search_dp
from environment import compute_simple_tpot, create_random_config


class DQNv1Trainer(BaseTrainer):
    """Trainer for V1: DQN + sum-based TPOT DP."""

    def __init__(self, config):
        super().__init__(config, "V1-DQN", max_minutes=config.max_training_minutes)
        self.cfg = config

    def _build_network(self):
        return DQNv1Network(hidden_dim=self.cfg.v1_hidden_dim).to(self.device)

    def _inference_one(self, network, devices, layers, ts, nl, nd):
        partition, tpot = dqn_v1_inference(
            network, devices, layers, ts, nl, nd, self.device, num_candidates=5
        )
        return partition, tpot

    def train(self):
        self.start_time = time.time()
        network = self._build_network()
        target_network = DQNv1Network(hidden_dim=self.cfg.v1_hidden_dim).to(self.device)
        target_network.load_state_dict(network.state_dict())

        optimizer = torch.optim.Adam(network.parameters(), lr=self.cfg.v1_learning_rate)
        buffer = ReplayBuffer(self.cfg.v1_replay_buffer_size)

        total_eps = self.cfg.v1_num_episodes
        batch_size = self.cfg.v1_batch_size
        gamma = self.cfg.v1_gamma
        eps_start = self.cfg.v1_epsilon_start
        eps_end = self.cfg.v1_epsilon_end
        eps_decay = self.cfg.v1_epsilon_decay
        target_update = self.cfg.v1_target_update_freq

        network.train()
        print(f"  [V1-DQN] Starting training: {total_eps} episodes")

        for ep in range(1, total_eps + 1):
            if self._check_time_budget():
                print(f"  [V1-DQN] Time budget reached at episode {ep}")
                break

            # Random environment
            devices, layers, ts = self._create_random_env(seed=ep)
            nl, nd = layers.num_layers, devices.num_devices

            # Epsilon schedule
            epsilon = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay)

            # Generate episode
            transitions, ordering, partition, tpot = dqn_v1_generate_episode(
                network, devices, layers, ts, nl, nd, self.device, epsilon=epsilon
            )

            # Compute reward
            reward = compute_reward(partition, devices, layers, ts)

            # Store transitions in buffer with final reward propagated
            # Apply reward to last transition
            for i, (obs, action, r, next_obs, done, h_state) in enumerate(transitions):
                actual_reward = reward if (i == len(transitions) - 1) else 0.0
                actual_done = done or (i == len(transitions) - 1)
                buffer.push(obs, action, actual_reward, next_obs, actual_done, h_state)

            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_tpot.append(tpot)

            # DQN update
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones, h_states = buffer.sample(batch_size)

                states_t = torch.FloatTensor(states).to(self.device)
                actions_t = torch.LongTensor(actions).to(self.device)
                rewards_t = torch.FloatTensor(rewards).to(self.device)
                next_states_t = torch.FloatTensor(next_states).to(self.device)
                dones_t = torch.FloatTensor(dones).to(self.device)
                h_states_t = torch.FloatTensor(h_states).to(self.device)

                # Current Q values (use stored hidden states)
                q_values, _ = network(states_t, h_states_t)
                q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                # Target Q values (Double DQN, use stored hidden states for next states too)
                with torch.no_grad():
                    next_q_online, _ = network(next_states_t, h_states_t)
                    next_actions = next_q_online.argmax(dim=1)

                    next_q_target, _ = target_network(next_states_t, h_states_t)
                    next_q_max = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

                    target_q = rewards_t + gamma * next_q_max * (1 - dones_t)

                loss = F.smooth_l1_loss(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), self.cfg.v1_max_grad_norm)
                optimizer.step()

                self.metrics.policy_losses.append(loss.item())

            # Update target network
            if ep % target_update == 0:
                target_network.load_state_dict(network.state_dict())

            # Evaluation
            if ep % self.cfg.eval_interval == 0:
                self._evaluate(network, ep)
                self._log_progress(ep, total_eps)

            self.metrics.wall_time.append(time.time() - self.start_time)

        # Final eval
        self._evaluate(network, total_eps)
        self._log_progress(total_eps, total_eps, "(final)")

        # Restore best
        self._restore_best(network)
        return network, self.metrics
