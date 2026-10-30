"""
PPO Training Pipeline for TrafficSignalEnv.

Trains a Proximal Policy Optimization agent using Stable-Baselines3.
Designed to run on CPU without a GPU.

Usage (from repo root):
    python -m backend.rl.train
    python -m backend.rl.train --timesteps 50000 --n-envs 1

Progress is logged every 1000 steps and checkpoints are saved every 10000 steps.
The trained model is written to backend/models/rl/ppo_traffic_latest.zip.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from backend.rl.config import EnvConfig
from backend.rl.environment import TrafficSignalEnv
from backend.rl.model_manager import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Hyperparameters (tuned for CPU training on a 4-way intersection)
# ──────────────────────────────────────────────────────────

DEFAULT_PPO_HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,          # Steps per environment per update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,            # Discount factor
    gae_lambda=0.95,       # GAE lambda
    clip_range=0.2,        # PPO clip range
    ent_coef=0.01,         # Entropy coefficient (encourages exploration)
    vf_coef=0.5,           # Value function coefficient
    max_grad_norm=0.5,
    verbose=1,
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  # 2-layer MLP
    ),
)


# ──────────────────────────────────────────────────────────
# Custom callback: logs per-episode statistics
# ──────────────────────────────────────────────────────────

class TrafficMetricsCallback(BaseCallback):
    """
    Logs per-episode reward, queue, and throughput statistics during training.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_rewards: list = []
        self._episode_lengths: list = []

    def _on_step(self) -> bool:
        # Collect episode info from Monitor wrapper
        if "episode" in self.locals.get("infos", [{}])[0]:
            ep_info = self.locals["infos"][0]["episode"]
            self._episode_rewards.append(ep_info.get("r", 0.0))
            self._episode_lengths.append(ep_info.get("l", 0))

        if self.num_timesteps % self.log_freq == 0 and self._episode_rewards:
            mean_r = float(np.mean(self._episode_rewards[-20:]))
            mean_l = float(np.mean(self._episode_lengths[-20:]))
            logger.info(
                f"[step {self.num_timesteps:>7d}] "
                f"mean_ep_reward={mean_r:+.3f}  mean_ep_length={mean_l:.0f}"
            )
        return True


# ──────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────

def make_env(episode_duration_sec: int = 900, seed: int = 0, simulator: str = "auto", sumo_gui: bool = False):
    """Factory function returning a Monitor-wrapped TrafficSignalEnv."""
    def _init():
        config = EnvConfig(
            episode_duration_sec=episode_duration_sec,
            simulator=simulator,
            sumo_gui=sumo_gui,
        )
        env = TrafficSignalEnv(config=config)
        env = Monitor(env)
        return env
    return _init


# ──────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────

def train(
    total_timesteps: int = 100_000,
    episode_duration_sec: int = 900,
    n_envs: int = 1,
    checkpoint_freq: int = 10_000,
    seed: int = 42,
    resume: bool = False,
    simulator: str = "auto",
    sumo_gui: bool = False,
    hyperparams: Optional[dict] = None,
) -> PPO:
    """
    Train a PPO agent on TrafficSignalEnv.

    Args:
        total_timesteps:      Total training timesteps.
        episode_duration_sec: Episode length in simulated seconds.
        n_envs:               Number of parallel environments (CPU).
        checkpoint_freq:      Save a checkpoint every N timesteps.
        seed:                 Random seed for reproducibility.
        resume:               If True, continue from latest saved model.
        simulator:            Simulator backend: 'auto' | 'junction_sim' | 'sumo'.
        sumo_gui:             Open SUMO GUI visualizer if using SUMO.
        hyperparams:          Override default PPO hyperparameters.

    Returns:
        Trained PPO model.
    """
    logger.info("=" * 60)
    logger.info("AI2 — PPO Training Pipeline")
    logger.info(f"  total_timesteps   : {total_timesteps:,}")
    logger.info(f"  episode_duration  : {episode_duration_sec}s")
    logger.info(f"  simulator         : {simulator}")
    logger.info(f"  n_envs            : {n_envs}")
    logger.info(f"  seed              : {seed}")
    logger.info(f"  resume            : {resume}")
    logger.info("=" * 60)

    # Build vectorised environment
    env_fns = [
        make_env(episode_duration_sec, seed + i, simulator=simulator, sumo_gui=sumo_gui)
        for i in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    # Observation normalisation (running mean/std) — helps PPO convergence
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # Merge hyperparams
    hp = {**DEFAULT_PPO_HYPERPARAMS, **(hyperparams or {})}

    if resume and model_manager.has_trained_model():
        logger.info(f"Resuming from {model_manager.latest_model_zip}")
        model = PPO.load(
            str(model_manager.latest_model_path),
            env=vec_env,
            **{k: v for k, v in hp.items() if k not in ("verbose",)},
        )
        model.set_random_seed(seed)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            **hp,
        )

    # Callbacks
    checkpoint_dir = str(model_manager.model_dir)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="ppo_checkpoint",
        save_vecnormalize=True,
        verbose=1,
    )
    metrics_cb = TrafficMetricsCallback(log_freq=1000, verbose=1)

    # Train
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, metrics_cb],
        reset_num_timesteps=not resume,
        progress_bar=False,
    )
    elapsed = time.time() - start

    # Save final model and metadata
    metadata = {
        "total_timesteps": model.num_timesteps,
        "episode_duration_sec": episode_duration_sec,
        "simulator": simulator,
        "environment_type": "sumo" if simulator == "sumo" else "junction_sim",
        "n_envs": n_envs,
        "seed": seed,
        "training_time_sec": round(elapsed, 1),
        "n_episodes_completed": len(metrics_cb._episode_rewards),
        "mean_ep_reward_last20": round(
            float(np.mean(metrics_cb._episode_rewards[-20:]))
            if metrics_cb._episode_rewards else 0.0,
            4,
        ),
        "hyperparams": {k: str(v) for k, v in hp.items()},
    }
    model_manager.save(model, metadata=metadata)

    logger.info(f"Training complete in {elapsed:.1f}s — model saved to {model_manager.latest_model_zip}")
    return model


# ──────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for AI2 Adaptive Traffic Signal Optimization"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--episode-duration", type=int, default=900,
        help="Episode length in simulated seconds (default: 900)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=1,
        help="Number of parallel envs (default: 1)"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=10_000,
        help="Checkpoint save frequency in timesteps (default: 10000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--simulator", type=str, default="auto", choices=["auto", "junction_sim", "sumo"],
        help="Simulation backend: 'auto' (detect SUMO, fallback to Python engine), 'junction_sim', or 'sumo'"
    )
    parser.add_argument(
        "--sumo-gui", action="store_true",
        help="Launch SUMO graphical visualization interface (requires sumo-gui)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from latest saved model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        total_timesteps=args.timesteps,
        episode_duration_sec=args.episode_duration,
        n_envs=args.n_envs,
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed,
        resume=args.resume,
        simulator=args.simulator,
        sumo_gui=args.sumo_gui,
    )

