"""Unified training entry point for all 6 versions."""
import time
import torch
import numpy as np

from config import TrainConfig
from training.train_v1 import DQNv1Trainer
from training.train_v2 import PPOv2Trainer
from training.train_v3 import PPOv3Trainer
from training.train_v4 import PPOv4Trainer
from training.train_v5 import MaskablePPOv5Trainer
from training.train_v6 import PPOv6Trainer
from training.train_v7 import PPOv7Trainer


def train_all(config: TrainConfig):
    """Train all selected versions sequentially.

    Returns:
        dict mapping version name -> (network, metrics)
    """
    results = {}

    if config.train_v1:
        print("\n" + "=" * 60)
        print("Training V1: DQN + min-max DP")
        print("=" * 60)
        trainer = DQNv1Trainer(config)
        net, metrics = trainer.train()
        results['v1'] = (net, metrics)

    if config.train_v2:
        print("\n" + "=" * 60)
        print("Training V2: PPO Binary + min-max DP")
        print("=" * 60)
        trainer = PPOv2Trainer(config)
        net, metrics = trainer.train()
        results['v2'] = (net, metrics)

    if config.train_v3:
        print("\n" + "=" * 60)
        print("Training V3: PPO-Clip One-Shot + min-max DP")
        print("=" * 60)
        trainer = PPOv3Trainer(config)
        net, metrics = trainer.train()
        results['v3'] = (net, metrics)

    if config.train_v4:
        print("\n" + "=" * 60)
        print("Training V4: PPO-Clip AutoReg + min-max DP")
        print("=" * 60)
        trainer = PPOv4Trainer(config)
        net, metrics = trainer.train()
        results['v4'] = (net, metrics)

    if config.train_v5:
        print("\n" + "=" * 60)
        print("Training V5: Maskable PPO-Clip + min-max DP")
        print("=" * 60)
        trainer = MaskablePPOv5Trainer(config)
        net, metrics = trainer.train()
        results['v5'] = (net, metrics)

    if config.train_v6:
        print("\n" + "=" * 60)
        print("Training V6: GNN-Based PPO + min-max DP")
        print("=" * 60)
        trainer = PPOv6Trainer(config)
        net, metrics = trainer.train()
        results['v6'] = (net, metrics)

    if config.train_v7:
        print("\n" + "=" * 60)
        print("Training V7: Autoregressive GNN-PPO + Positional Encoding")
        print("=" * 60)
        trainer = PPOv7Trainer(config)
        net, metrics = trainer.train()
        results['v7'] = (net, metrics)

    return results
