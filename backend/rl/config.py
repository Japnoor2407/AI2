"""
Reinforcement Learning configuration for TrafficSignalEnv.
Defines observation bounds, action mappings, reward weights, and episode defaults.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# Allowed phase definitions for 2-phase intersection control
# Phase 1: North-South arterial green
# Phase 2: East-West crossway green
SUPPORTED_PHASES: List[int] = [1, 2]

# Allowed discrete green durations in seconds
GREEN_DURATIONS: List[int] = [10, 15, 20, 25, 30, 40, 50, 55]

# Clearance intervals (seconds)
YELLOW_CLEARANCE_SEC: int = 3
ALL_RED_CLEARANCE_SEC: int = 2

# Normalization constants
MAX_QUEUE_PER_APPROACH: float = 50.0   # PCU
MAX_ARRIVAL_RATE: float = 60.0         # PCU/min
MAX_WAIT_TIME_SEC: float = 120.0       # seconds
MAX_TIME_SINCE_SERVED: float = 120.0   # seconds
MAX_TOTAL_QUEUE: float = 200.0         # PCU
STARVATION_THRESHOLD_SEC: float = 80.0 # seconds


@dataclass
class RewardWeights:
    """Weights for the multi-objective reward function."""
    queue_improvement: float = 0.35
    throughput: float = 0.30
    waiting_time: float = 0.20
    starvation: float = 0.10
    switching: float = 0.05
    balance: float = 0.05


@dataclass
class EnvConfig:
    """Configuration for TrafficSignalEnv."""
    junction_id: str = "AlankarChowk"
    date: str = "11"
    plan_mode: str = "2_phase"  # 2_phase (NS / EW)
    episode_duration_sec: int = 900  # 15 minutes of simulation per episode
    min_green_sec: int = 10
    max_green_sec: int = 55
    yellow_sec: int = YELLOW_CLEARANCE_SEC
    all_red_sec: int = ALL_RED_CLEARANCE_SEC
    starvation_threshold_sec: float = STARVATION_THRESHOLD_SEC
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    render_mode: str = "human"
    simulator: str = "auto"  # "auto" | "junction_sim" | "sumo"
    sumo_gui: bool = False

