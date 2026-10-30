"""
Action space definition for TrafficSignalEnv.

Implements a 16-element discrete action space:
2 phases (Phase 1: North-South, Phase 2: East-West)
x 8 green durations (10s, 15s, 20s, 25s, 30s, 40s, 50s, 55s)
= 16 discrete actions.
"""

from typing import Dict, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from backend.rl.config import SUPPORTED_PHASES, GREEN_DURATIONS


class ActionDecoder:
    """
    Decodes and encodes discrete RL actions into phase targets and green durations.
    """

    def __init__(
        self,
        phases: List[int] = SUPPORTED_PHASES,
        durations: List[int] = GREEN_DURATIONS,
    ):
        self.phases = list(phases)
        self.durations = list(durations)
        self.num_phases = len(self.phases)
        self.num_durations = len(self.durations)
        self.total_actions = self.num_phases * self.num_durations

        # Prebuild mapping tables
        self._action_table: List[Tuple[int, int]] = []
        self._inverse_table: Dict[Tuple[int, int], int] = {}

        for p in self.phases:
            for d in self.durations:
                idx = len(self._action_table)
                self._action_table.append((p, d))
                self._inverse_table[(p, d)] = idx

    def get_space(self) -> spaces.Discrete:
        """Return the Gymnasium Discrete action space."""
        return spaces.Discrete(self.total_actions)

    def decode(self, action: int) -> Tuple[int, int]:
        """
        Convert discrete action index to (target_phase_id, green_duration_sec).
        Raises ValueError if action is out of range.
        """
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif isinstance(action, (int, np.integer)):
            action = int(action)
        else:
            try:
                action = int(action)
            except Exception:
                raise ValueError(f"Action {action} cannot be converted to integer.")

        if action < 0 or action >= self.total_actions:
            raise ValueError(
                f"Action {action} is invalid. Must be in range [0, {self.total_actions - 1}]."
            )
        return self._action_table[action]

    def encode(self, phase_id: int, duration_sec: int) -> int:
        """
        Convert (phase_id, duration_sec) to discrete action index.
        Finds the closest supported green duration if duration is not exact.
        """
        if phase_id not in self.phases:
            phase_id = self.phases[0]

        # Snap duration to closest allowed
        closest_duration = min(self.durations, key=lambda d: abs(d - duration_sec))
        return self._inverse_table[(phase_id, closest_duration)]

    def describe(self, action: int) -> str:
        """Return human-readable description of an action."""
        phase_id, duration = self.decode(action)
        phase_desc = "North-South" if phase_id == 1 else "East-West"
        return f"Phase {phase_id} ({phase_desc}) for {duration}s"
