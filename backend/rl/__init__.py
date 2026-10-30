"""
Reinforcement Learning package for AI2 Adaptive Traffic Signal Optimization.
Provides Gymnasium environment, observation/action spaces, reward calculator,
PPO training pipeline, inference engine, and model management.
"""

from backend.rl.config import EnvConfig, RewardWeights
from backend.rl.action_space import ActionDecoder
from backend.rl.observation import ObservationBuilder, OBSERVATION_DIM, OBSERVATION_FEATURE_NAMES
from backend.rl.reward import RewardCalculator
from backend.rl.environment import TrafficSignalEnv
from backend.rl.sumo_adapter import SimulationAdapter, is_sumo_available
from backend.rl.model_manager import ModelManager, model_manager
from backend.rl.inference import RLInferenceEngine, rl_inference

__all__ = [
    "EnvConfig",
    "RewardWeights",
    "ActionDecoder",
    "ObservationBuilder",
    "OBSERVATION_DIM",
    "OBSERVATION_FEATURE_NAMES",
    "RewardCalculator",
    "TrafficSignalEnv",
    "SimulationAdapter",
    "is_sumo_available",
    "ModelManager",
    "model_manager",
    "RLInferenceEngine",
    "rl_inference",
]

