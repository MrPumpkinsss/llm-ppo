# Read the file
with open('train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find PPOv1Trainer start and PPOv2Trainer start
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if 'class PPOv1Trainer:' in line:
        start_idx = i
    if start_idx is not None and 'class PPOv2Trainer:' in line:
        end_idx = i
        break

print(f"Found PPOv1Trainer at line {start_idx+1}, PPOv2Trainer at line {end_idx+1}")

new_class = """
class PPOv1Trainer:
    \"\"\"Trainer for PPO-v1: supervised pre-training on DP partitions + PPO fine-tuning.\"\"\"

    def __init__(self, config):
        self.config = config
        self.max_devices = 10
        self.max_layers = 64
        self.max_obs_dim = get_obs_dim(self.max_devices, self.max_layers)

        self.network = DirectAssignNetwork(
            num_devices=self.max_devices,
            obs_dim=self.max_obs_dim,
            hidden_dim=config.v1_hidden_dim,
        ).to(config.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.v1_learning_rate)
        self.metrics = TrainingMetrics()

    def _sample_partition(self, obs_t, num_layers, num_devices, device, greedy=True):
        with torch.no_grad():
            logits, _ = self.network(obs_t, num_layers)
        logits = logits.squeeze(0)
        log_probs, entropies, partition, max_dev = [], [], [], 0
        for i in range(num_layers):
            mask = torch.full((num_devices,), -1e9, device=device)
            mask[max_dev:] = 0.0
            layer_logits = logits[i] + mask
            probs = F.softmax(layer_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = probs.argmax() if greedy else dist.sample()
            partition.append(action.item())
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            if action.item() > max_dev: max_dev = action.item()
        total_lp = sum(log_probs) if log_probs else torch.tensor(0.0, device=device)
        total_ent = sum(entropies) if entropies else torch.tensor(0.0, device=device)
        return partition, total_lp, total_ent

    def _log_prob_partition(self, obs_t, partition, num_layers, num_devices, device):
        with torch.no_grad():
            logits, _ = self.network(obs_t, num_layers)
        logits = logits.squeeze(0)
        total_lp = torch.tensor(0.0, device=device)
        max_dev = 0
        for i in range(num_layers):
            mask = torch.full((num_devices,), -1e9, device=device)
            mask[max_dev:] = 0.0
            layer_logits = logits[i] + mask
            probs = F.softmax(layer_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            total_lp += dist.log_prob(torch.tensor(partition[i], device=device))
            max_dev = max(max_dev, partition[i])
        return total_lp

    def train(self, num_episodes=None):
        num_episodes = num_episodes or self.config.v1_num_episodes
        device = self.config.device
        start_time = time.time()

        print(\"[PPO-v1] Supervised pre-training on DP-optimal partitions...\")
        pretrain_episodes = min(1000, num_episodes // 2)
        pretrain_opt = optim.Adam(self.network.parameters(), lr=self.config.v1_learning_rate)

        for episode in range(pretrain_episodes):
            seed = self.config.seed + episode * 17
            num_layers = np.random.choice([16, 24, 32, 40, 48, 56, 64])
            num_devices = np.random.randint(2, 11)
            devs, lys, ts = create_random_config(num_layers, num_devices, seed=seed)

            dp_part = dp_partition(num_layers, num_devices, devs, lys, ts)
            targets = torch.tensor(dp_part, device=device, dtype=torch.long)

            obs_full = build_observation(devs, lys, num_devices, num_layers)
            obs_padded = np.zeros(self.max_obs_dim, dtype=np.float32)
            obs_padded[:get_obs_dim(num_devices, num_layers)] = obs_full
            obs_t = torch.tensor(obs_padded, device=device).unsqueeze(0)

            logits, _ = self.network(obs_t, num_layers)
            logits = logits.squeeze(0)

            total_loss = torch.tensor(0.0, device=device)
            max_dev = 0
            for i in range(num_layers):
                mask = torch.full((num_devices,), -1e9, device=device)
                mask[max_dev:] = 0.0
                masked = logits[i] + mask
                total_loss += F.cross_entropy(masked.unsqueeze(0), targets[i].unsqueeze(0))
                max_dev = max(max_dev, dp_part[i])

            (total_loss / num_layers).backward()
            pretrain_opt.step()
            pretrain_opt.zero_grad()

            if (episode + 1) % 200 == 0:
                print(f\"  [Pre {episode+1}/{pretrain_episodes}] Loss: {total_loss.item()/num_layers:.4f}\")

        print(\"[PPO-v1] RL fine-tuning...\")
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.v1_learning_rate * 0.3)

        obs_buf, part_buf, lp_buf = [], [], []
        rew_buf, ent_buf, cfg_buf = [], [], []

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

            partition, lp, ent = self._sample_partition(obs_t, num_layers, num_devices, device, greedy=False)
            reward = -compute_simple_tpot(partition, devs, lys, ts)

            if -reward <= dp_tpot * 1.01: reward += 10.0
            elif -reward <= dp_tpot * 1.05: reward += 5.0
            elif -reward <= dp_tpot * 1.10: reward += 2.0
            tpot = -reward

            obs_buf.append(obs_padded)
            part_buf.append(partition)
            lp_buf.append(lp.item())
            rew_buf.append(reward)
            ent_buf.append(ent.item())
            cfg_buf.append((num_layers, num_devices))

            self.metrics.episode_rewards.append(reward)
            self.metrics.episode_tpot.append(tpot)
            self.metrics.wall_time.append(time.time() - start_time)

            if len(obs_buf) >= self.config.v1_batch_size:
                self._ppo_update(obs_buf, part_buf, lp_buf, rew_buf, ent_buf, cfg_buf)
                obs_buf.clear(); part_buf.clear(); lp_buf.clear()
                rew_buf.clear(); ent_buf.clear(); cfg_buf.clear()

            if (episode + 1) % self.config.eval_interval == 0:
                self._evaluate(episode + 1)

            elapsed = time.time() - start_time
            if elapsed > self.config.max_training_minutes * 60:
                print(f\"[PPO-v1] Time limit ({elapsed/60:.1f}m)\")
                break
            if (episode + 1) % 200 == 0:
                print(f\"[PPO-v1] Ep {episode+1}/{num_episodes} | TPOT: {np.mean(self.metrics.episode_tpot[-200:]):.4f} | {time.time()-start_time:.0f}s\")

        return self.network, self.metrics

    def _ppo_update(self, obs_l, part_l, lp_l, rew_l, ent_l, cfg_l):
        device = self.config.device
        B = len(obs_l)
        obs_t = torch.tensor(np.array(obs_l), device=device)
        old_lp = torch.tensor(lp_l, device=device)
        rewards = torch.tensor(rew_l, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for _ in range(self.config.v1_epochs_per_update):
            nlp, val = [], []
            for i in range(B):
                nl, nd = cfg_l[i]
                obs_i = obs_t[i:i+1]
                lp = self._log_prob_partition(obs_i, part_l[i], nl, nd, device)
                _, v = self.network(obs_i, nl)
                nlp.append(lp); val.append(v.squeeze())

            nlp = torch.stack(nlp); val = torch.stack(val)
            ratio = torch.exp(nlp - old_lp)
            ploss = -torch.min(ratio * rewards, torch.clamp(ratio, 1-self.config.v1_clip_eps, 1+self.config.v1_clip_eps) * rewards).mean()
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
            p, _, _ = self._sample_partition(ot, nl, nd, device, greedy=True)
            ppo_l.append(compute_simple_tpot(p, devs, lys, ts))
            dp_l.append(compute_simple_tpot(dp_partition(nl, nd, devs, lys, ts), devs, lys, ts))
            gr_l.append(compute_simple_tpot(greedy_partition_advanced(nl, nd, devs, lys, ts), devs, lys, ts))
        self.metrics.eval_tpot.append(np.mean(ppo_l))
        self.metrics.dp_tpot.append(np.mean(dp_l))
        self.metrics.greedy_tpot.append(np.mean(gr_l))
        print(f\"[PPO-v1 Eval @{step}] PPO: {np.mean(ppo_l):.4f} | DP: {np.mean(dp_l):.4f} | Greedy: {np.mean(gr_l):.4f}\")

"""

new_lines = lines[:start_idx] + [new_class] + lines[end_idx:]
with open('train.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Done! File updated.")
