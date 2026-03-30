"""
train.py
"""

import os, json, time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env import LLMPartitionEnv, FixedClusterEnv, MAX_DEVICES


class MetricsCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=500, n_eval=50, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.best_tpot = float('inf')
        self.stag = 0
        self.best_path = None
        self.metrics = {k: [] for k in
                        ['timesteps', 'mean_tpot', 'min_tpot', 'std_tpot',
                         'mean_reward', 'ent_coef', 'mean_switches']}

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            tpots, rews, switches = [], [], []
            for ep in range(self.n_eval):
                obs, _ = self.eval_env.reset(seed=ep + 1000)
                act, _ = self.model.predict(obs, deterministic=True)
                _, rew, _, _, info = self.eval_env.step(act)
                if 'tpot' in info:
                    tpots.append(info['tpot'])
                    switches.append(info.get('num_device_switches', 0))
                rews.append(rew)

            if tpots:
                mt = np.mean(tpots)
                self.metrics['timesteps'].append(self.num_timesteps)
                self.metrics['mean_tpot'].append(float(mt))
                self.metrics['min_tpot'].append(float(np.min(tpots)))
                self.metrics['std_tpot'].append(float(np.std(tpots)))
                self.metrics['mean_reward'].append(float(np.mean(rews)))
                self.metrics['ent_coef'].append(float(self.model.ent_coef))
                self.metrics['mean_switches'].append(float(np.mean(switches)))

                if mt < self.best_tpot - 0.1:
                    self.best_tpot = mt; self.stag = 0
                    if self.best_path: self.model.save(self.best_path)
                else:
                    self.stag += 1

                if self.stag > 10:
                    old = self.model.ent_coef
                    self.model.ent_coef = min(0.3, old * 1.5)
                    self.stag = 0
                    if self.verbose:
                        print(f"\n  ⚡ Stag! ent: {old:.4f}→{self.model.ent_coef:.4f}")
                elif self.stag == 0:
                    self.model.ent_coef = max(0.001, self.model.ent_coef * 0.98)

                if self.verbose:
                    print(f"\n  [Eval@{self.num_timesteps}] mean={mt:.1f} min={np.min(tpots):.1f} "
                          f"best={self.best_tpot:.1f} ent={self.model.ent_coef:.4f} "
                          f"sw={np.mean(switches):.1f}")
        return True


def train(config=None):
    if config is None:
        config = {
            'total_timesteps': 1_000_000,
            'n_envs': 16,
            'learning_rate': 3e-4,
            'n_steps': 128,       # 单步env，128 episodes per update
            'batch_size': 64,
            'n_epochs': 15,
            'gamma': 1.0,        # 单步 → gamma无影响
            'gae_lambda': 0.95,
            'ent_coef': 0.05,
            'clip_range': 0.2,
            'eval_freq': 500,
            'save_dir': 'results',
        }
    os.makedirs(config['save_dir'], exist_ok=True)
    print("=" * 60)
    print("Training: 28L, cut-point action space")
    print("=" * 60)

    fixed = [(2,42),(3,42),(3,123),(4,42),(4,789),(5,42),(5,456),(6,42)]
    fns = []
    nf = config['n_envs'] // 2
    for i in range(nf):
        nd, cs = fixed[i % len(fixed)]
        fns.append(lambda nd=nd, cs=cs, i=i: Monitor(FixedClusterEnv(nd, cs, seed=i*100)))
    for i in range(config['n_envs'] - nf):
        fns.append(lambda i=i: Monitor(LLMPartitionEnv(seed=i*100+50)))

    train_env = DummyVecEnv(fns)
    eval_env = FixedClusterEnv(num_devices=4, cluster_seed=42)

    print(f"Obs: {train_env.observation_space.shape}")
    print(f"Act: {train_env.action_space.shape}")

    model = PPO("MlpPolicy", train_env,
                learning_rate=config['learning_rate'],
                n_steps=config['n_steps'],
                batch_size=config['batch_size'],
                n_epochs=config['n_epochs'],
                gamma=config['gamma'],
                gae_lambda=config['gae_lambda'],
                ent_coef=config['ent_coef'],
                clip_range=config['clip_range'],
                max_grad_norm=0.5, verbose=1, device='cpu', seed=42,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                    activation_fn=torch.nn.Tanh,
                    log_std_init=-1.0,  # 初始较小的std，让探索适度
                ))
    print(model.policy)

    mp = os.path.join(config['save_dir'], 'ppo_llm_partition')
    cb = MetricsCallback(eval_env, config['eval_freq'])
    cb.best_path = mp + '_best'

    t0 = time.time()
    model.learn(config['total_timesteps'], callback=cb, progress_bar=True)
    print(f"\nDone in {time.time()-t0:.1f}s")
    model.save(mp)

    for name, data in [('training_metrics.json', cb.metrics), ('config.json', config)]:
        with open(os.path.join(config['save_dir'], name), 'w') as f:
            json.dump(data, f, indent=2)
    return model, cb.metrics, config


if __name__ == "__main__":
    train()