"""
Observation space definition and state vector builder for TrafficSignalEnv.

Produces a normalized 21-dimensional float32 vector representing the intersection's
current queue state, arrivals, waiting times, starvation status, signal state, and time.
"""

from typing import Dict, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from backend.core.schema import ApproachState
from backend.rl.config import (
    MAX_QUEUE_PER_APPROACH,
    MAX_ARRIVAL_RATE,
    MAX_WAIT_TIME_SEC,
    MAX_TIME_SINCE_SERVED,
    MAX_TOTAL_QUEUE,
)


OBSERVATION_FEATURE_NAMES = [
    # Queues (4)
    "queue_north",
    "queue_south",
    "queue_east",
    "queue_west",
    # Arrival rates (4)
    "arrival_rate_north",
    "arrival_rate_south",
    "arrival_rate_east",
    "arrival_rate_west",
    # Average wait time (4)
    "avg_wait_north",
    "avg_wait_south",
    "avg_wait_east",
    "avg_wait_west",
    # Starvation timers (4)
    "time_since_served_north",
    "time_since_served_south",
    "time_since_served_east",
    "time_since_served_west",
    # Current active phase one-hot (2)
    "phase_1_active",
    "phase_2_active",
    # Elapsed duration in active phase (1)
    "elapsed_in_phase_norm",
    # Total intersection density (1)
    "total_queue_density_norm",
    # Episode / time progress (1)
    "simulation_progress_norm",
]

OBSERVATION_DIM = len(OBSERVATION_FEATURE_NAMES)  # 21


class ObservationBuilder:
    """
    Constructs normalized, finite float32 observation vectors from simulation telemetry.
    """

    def __init__(self):
        self.feature_names = list(OBSERVATION_FEATURE_NAMES)
        self.dimension = OBSERVATION_DIM

    def get_space(self) -> spaces.Box:
        """Return the Gymnasium Box observation space."""
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.dimension,),
            dtype=np.float32,
        )

    def build(
        self,
        approaches: Dict[str, ApproachState],
        active_phase_id: int,
        elapsed_in_phase: int,
        current_step: int,
        total_steps: int,
    ) -> np.ndarray:
        """
        Extract and normalize features from simulation approach states into a Box vector.
        Guarantees that observations are within [0.0, 1.0] and contain no NaN or inf values.
        """
        # Map approach identifiers to standardized directional slots
        app_by_dir: Dict[str, ApproachState] = {}
        for app in approaches.values():
            dir_code = (app.direction or "").upper()
            if "UP" in dir_code or "NORTH" in app.approach_id.upper():
                app_by_dir["N"] = app
            elif "DOWN" in dir_code or "SOUTH" in app.approach_id.upper():
                app_by_dir["S"] = app
            elif "RIGHT" in dir_code or "EAST" in app.approach_id.upper():
                app_by_dir["E"] = app
            elif "LEFT" in dir_code or "WEST" in app.approach_id.upper():
                app_by_dir["W"] = app

        # Fallback to key order if directions were not found
        keys = sorted(approaches.keys())
        if "N" not in app_by_dir and len(keys) > 0:
            app_by_dir["N"] = approaches[keys[0]]
        if "S" not in app_by_dir and len(keys) > 1:
            app_by_dir["S"] = approaches[keys[1]]
        if "E" not in app_by_dir and len(keys) > 2:
            app_by_dir["E"] = approaches[keys[2]]
        if "W" not in app_by_dir and len(keys) > 3:
            app_by_dir["W"] = approaches[keys[3]]

        def _safe_val(app: Optional[ApproachState], attr: str, max_val: float) -> float:
            if app is None:
                return 0.0
            val = getattr(app, attr, 0.0)
            if not np.isfinite(val):
                return 0.0
            return float(np.clip(val / max_val, 0.0, 1.0))

        # 1-4: Queues
        q_n = _safe_val(app_by_dir.get("N"), "estimated_queue_pcu", MAX_QUEUE_PER_APPROACH)
        q_s = _safe_val(app_by_dir.get("S"), "estimated_queue_pcu", MAX_QUEUE_PER_APPROACH)
        q_e = _safe_val(app_by_dir.get("E"), "estimated_queue_pcu", MAX_QUEUE_PER_APPROACH)
        q_w = _safe_val(app_by_dir.get("W"), "estimated_queue_pcu", MAX_QUEUE_PER_APPROACH)

        # 5-8: Arrival rates
        arr_n = _safe_val(app_by_dir.get("N"), "arrival_rate_pcu_per_min", MAX_ARRIVAL_RATE)
        arr_s = _safe_val(app_by_dir.get("S"), "arrival_rate_pcu_per_min", MAX_ARRIVAL_RATE)
        arr_e = _safe_val(app_by_dir.get("E"), "arrival_rate_pcu_per_min", MAX_ARRIVAL_RATE)
        arr_w = _safe_val(app_by_dir.get("W"), "arrival_rate_pcu_per_min", MAX_ARRIVAL_RATE)

        # 9-12: Average wait times
        w_n = _safe_val(app_by_dir.get("N"), "average_wait_time_sec", MAX_WAIT_TIME_SEC)
        w_s = _safe_val(app_by_dir.get("S"), "average_wait_time_sec", MAX_WAIT_TIME_SEC)
        w_e = _safe_val(app_by_dir.get("E"), "average_wait_time_sec", MAX_WAIT_TIME_SEC)
        w_w = _safe_val(app_by_dir.get("W"), "average_wait_time_sec", MAX_WAIT_TIME_SEC)

        # 13-16: Starvation timers (time since last served)
        ts_n = _safe_val(app_by_dir.get("N"), "time_since_last_served_sec", MAX_TIME_SINCE_SERVED)
        ts_s = _safe_val(app_by_dir.get("S"), "time_since_last_served_sec", MAX_TIME_SINCE_SERVED)
        ts_e = _safe_val(app_by_dir.get("E"), "time_since_last_served_sec", MAX_TIME_SINCE_SERVED)
        ts_w = _safe_val(app_by_dir.get("W"), "time_since_last_served_sec", MAX_TIME_SINCE_SERVED)

        # 17-18: Active phase one-hot (Phase 1 = NS, Phase 2 = EW)
        p1 = 1.0 if active_phase_id == 1 else 0.0
        p2 = 1.0 if active_phase_id == 2 else 0.0

        # 19: Elapsed in current phase
        elapsed_norm = float(np.clip(elapsed_in_phase / 60.0, 0.0, 1.0))

        # 20: Total junction density
        total_q = sum(getattr(a, "estimated_queue_pcu", 0.0) for a in approaches.values())
        density_norm = float(np.clip(total_q / MAX_TOTAL_QUEUE, 0.0, 1.0))

        # 21: Progress through episode
        progress = float(np.clip(current_step / max(1, total_steps), 0.0, 1.0))

        obs = np.array(
            [
                q_n, q_s, q_e, q_w,
                arr_n, arr_s, arr_e, arr_w,
                w_n, w_s, w_e, w_w,
                ts_n, ts_s, ts_e, ts_w,
                p1, p2,
                elapsed_norm,
                density_norm,
                progress,
            ],
            dtype=np.float32,
        )

        # Replace any non-finite with 0.0 just in case
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(obs, 0.0, 1.0)

