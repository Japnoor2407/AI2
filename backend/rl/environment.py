"""
Custom Gymnasium Environment for Adaptive Traffic Signal Optimization.

Implements TrafficSignalEnv adhering to Gymnasium standards.
Drives a four-way intersection using real Pune CCTV traffic arrival profiles
and the discrete-time JunctionSimulator engine.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from backend.core.schema import (
    ApproachState,
    ControllerDecision,
    ControllerMode,
    PhaseConfig,
    SignalState,
    StepObservation,
    VehicleClassCounts,
)
from backend.core.config import PCU_WEIGHTS
from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.replay_engine import JunctionSimulator
from backend.rl.sumo_adapter import SimulationAdapter
from backend.data.loader import DatasetAdapter
from backend.rl.config import EnvConfig, RewardWeights
from backend.rl.action_space import ActionDecoder
from backend.rl.observation import ObservationBuilder, OBSERVATION_DIM
from backend.rl.reward import RewardCalculator

logger = logging.getLogger(__name__)


class TrafficSignalEnv(gym.Env):
    """
    TrafficSignalEnv simulates an intelligent traffic signal controller.

    Observation space:
        Box(21,), float32 values in [0.0, 1.0] representing normalized queues,
        arrival rates, average delays, starvation timers, active phase, and episode progress.

    Action space:
        Discrete(16): Combination of 2 phases (NS, EW) and 8 green durations (10s to 55s).
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        arrival_data: Optional[pd.DataFrame] = None,
        dataset_adapter: Optional[DatasetAdapter] = None,
    ):
        super().__init__()
        self.config = config or EnvConfig()
        self._raw_arrival_data = arrival_data
        self.adapter = dataset_adapter or DatasetAdapter()

        # Layout (2-phase: Phase 1 = North-South, Phase 2 = East-West)
        self.layout = JunctionLayout(
            junction_id=self.config.junction_id,
            plan_mode=self.config.plan_mode,
        )

        # Helpers
        self.action_decoder = ActionDecoder(phases=[1, 2])
        self.observation_builder = ObservationBuilder()
        self.reward_calculator = RewardCalculator(weights=self.config.reward_weights)

        # Gymnasium Spaces
        self.action_space: spaces.Discrete = self.action_decoder.get_space()
        self.observation_space: spaces.Box = self.observation_builder.get_space()

        # Runtime state
        self.sim: Optional[SimulationAdapter] = None
        self.active_phase_id: int = 1
        self.elapsed_in_phase: int = 0
        self.episode_steps: int = 0
        self.max_steps: int = self.config.episode_duration_sec
        self.last_observation: Optional[np.ndarray] = None
        self.last_step_obs: Optional[StepObservation] = None
        self.total_episode_reward: float = 0.0

        # Load or generate baseline arrival data
        self._prepared_df = self._prepare_data()

    def _prepare_data(self) -> pd.DataFrame:
        """Load empirical dataset or generate a synthetic benchmark fallback."""
        if self._raw_arrival_data is not None and not self._raw_arrival_data.empty:
            return self._raw_arrival_data

        try:
            cameras = set(m["camera_id"] for m in self.layout.approaches.values())
            primary_cam = sorted(cameras)[0] if cameras else "a2"
            df = self.adapter.load_camera_arrivals(
                self.config.junction_id,
                primary_cam,
                self.config.date,
                max_seconds=self.config.episode_duration_sec + 200,
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Could not load empirical dataset: {e}. Generating synthetic fallback.")

        # Synthetic benchmark demand profile (clearly marked)
        return self._create_synthetic_arrivals()

    def _create_synthetic_arrivals(self) -> pd.DataFrame:
        """Generates realistic synthetic arrivals strictly for offline tests when dataset CSV is missing."""
        rows = []
        directions = ["UP", "DOWN", "RIGHT", "LEFT"]
        for s in range(self.config.episode_duration_sec + 100):
            # Time string
            total_sec = 9 * 3600 + s
            h = (total_sec // 3600) % 24
            m = (total_sec % 3600) // 60
            sec = total_sec % 60
            time_str = f"{h:02d}:{m:02d}:{sec:02d}"

            for d in directions:
                # Stochastic arrival rate
                base = 0.4 if d in ["UP", "DOWN"] else 0.25
                cars = 1 if np.random.rand() < base else 0
                bikes = 1 if np.random.rand() < (base * 1.5) else 0
                rows.append({
                    "Time": time_str,
                    "Direction": d,
                    "car_arrivals": cars,
                    "motorbike_arrivals": bikes,
                    "bus_arrivals": 0,
                    "truck_arrivals": 0,
                })
        return pd.DataFrame(rows)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets environment to starting state and returns (initial_observation, info).
        """
        super().reset(seed=seed)

        # Re-initialize simulation engine (SUMO or JunctionSimulator fallback)
        use_sumo = None
        if self.config.simulator == "sumo":
            use_sumo = True
        elif self.config.simulator == "junction_sim":
            use_sumo = False

        self.sim = SimulationAdapter(
            layout=self.layout,
            arrival_data=self._prepared_df,
            date=self.config.date,
            use_sumo=use_sumo,
            sumo_gui=self.config.sumo_gui,
        )
        self.active_phase_id = 1
        self.elapsed_in_phase = 0
        self.episode_steps = 0
        self.total_episode_reward = 0.0

        # Step 0 initial tick to populate approach queues and observation
        initial_decision = ControllerDecision(
            step=0,
            timestamp_str=self.sim.time_index[0] if self.sim.time_index else "09:00:00",
            mode=ControllerMode.ADAPTIVE,
            active_phase_id=self.active_phase_id,
            phase_name="Phase 1: North-South Arterial",
            signal_state=SignalState.GREEN,
            time_remaining_in_state=self.config.min_green_sec,
            elapsed_in_phase=0,
            decision_rationale="Initial episode state",
        )
        obs_step = self.sim.step(initial_decision)
        self.last_step_obs = obs_step

        obs_vector = self.observation_builder.build(
            approaches=self.sim.approaches,
            active_phase_id=self.active_phase_id,
            elapsed_in_phase=self.elapsed_in_phase,
            current_step=self.sim.step_count,
            total_steps=self.max_steps,
        )
        self.last_observation = obs_vector

        info = {
            "step": 0,
            "active_phase_id": self.active_phase_id,
            "total_queue_pcu": obs_step.total_active_queue_pcu,
            "average_delay_sec": obs_step.average_delay_sec,
            "total_throughput_pcu": 0.0,
            "starvation_occurrences": 0,
            "simulator_backend": self.sim.backend,
        }

        return obs_vector, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes discrete action: (target_phase_id, green_duration_sec).
        Safely enforces yellow transition (3s) and all-red clearance (2s)
        before green hold, accumulating rewards and simulation state across the interval.
        """
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        # Normalize action to scalar int
        if isinstance(action, np.ndarray):
            act_idx = int(action.item())
        else:
            act_idx = int(action)

        # Decode discrete action into phase and duration
        target_phase_id, requested_duration = self.action_decoder.decode(act_idx)

        # Snapshot approaches before action execution
        prev_approaches = {
            k: v.model_copy() for k, v in self.sim.approaches.items()
        }
        initial_throughput = self.sim._total_throughput_pcu
        phase_switched = (target_phase_id != self.active_phase_id)

        # ── 1. Safety Clearance Phase (if switching) ──────────────────────────
        if phase_switched:
            # 1a. Yellow clearance for old phase
            for _ in range(self.config.yellow_sec):
                if self.sim.step_count >= self.sim.total_steps or self.sim.step_count >= self.max_steps:
                    break
                ts = self.sim.time_index[self.sim.step_count] if self.sim.step_count < len(self.sim.time_index) else ""
                decision = ControllerDecision(
                    step=self.sim.step_count,
                    timestamp_str=ts,
                    mode=ControllerMode.ADAPTIVE,
                    active_phase_id=self.active_phase_id,
                    phase_name=f"Phase {self.active_phase_id} Yellow Clearance",
                    signal_state=SignalState.YELLOW,
                    time_remaining_in_state=self.config.yellow_sec,
                    elapsed_in_phase=self.elapsed_in_phase,
                    is_transitioning=True,
                    transition_stage="YELLOW",
                    decision_rationale="Yellow clearance interval",
                )
                self.sim.step(decision)
                self.elapsed_in_phase += 1

            # 1b. All-red clearance (all stopped)
            for _ in range(self.config.all_red_sec):
                if self.sim.step_count >= self.sim.total_steps or self.sim.step_count >= self.max_steps:
                    break
                ts = self.sim.time_index[self.sim.step_count] if self.sim.step_count < len(self.sim.time_index) else ""
                decision = ControllerDecision(
                    step=self.sim.step_count,
                    timestamp_str=ts,
                    mode=ControllerMode.ADAPTIVE,
                    active_phase_id=self.active_phase_id,
                    phase_name="All-Red Clearance",
                    signal_state=SignalState.ALL_RED,
                    time_remaining_in_state=self.config.all_red_sec,
                    elapsed_in_phase=self.elapsed_in_phase,
                    is_transitioning=True,
                    transition_stage="ALL_RED",
                    decision_rationale="Inter-green safety clearance",
                )
                self.sim.step(decision)
                self.elapsed_in_phase += 1

            # Switch active phase to target
            self.active_phase_id = target_phase_id
            self.elapsed_in_phase = 0

        # ── 2. Green Hold Phase ───────────────────────────────────────────────
        green_duration = int(np.clip(
            requested_duration,
            self.config.min_green_sec,
            self.config.max_green_sec,
        ))

        phase_name = (
            "Phase 1: North-South Arterial"
            if self.active_phase_id == 1
            else "Phase 2: East-West Crossway"
        )

        for sec in range(green_duration):
            if self.sim.step_count >= self.sim.total_steps or self.sim.step_count >= self.max_steps:
                break
            ts = self.sim.time_index[self.sim.step_count] if self.sim.step_count < len(self.sim.time_index) else ""
            rem = max(0, green_duration - sec)
            decision = ControllerDecision(
                step=self.sim.step_count,
                timestamp_str=ts,
                mode=ControllerMode.ADAPTIVE,
                active_phase_id=self.active_phase_id,
                phase_name=phase_name,
                signal_state=SignalState.GREEN,
                time_remaining_in_state=rem,
                elapsed_in_phase=self.elapsed_in_phase,
                is_transitioning=False,
                decision_rationale=f"RL Selected Green: {phase_name} ({green_duration}s)",
            )
            obs_step = self.sim.step(decision)
            self.last_step_obs = obs_step
            self.elapsed_in_phase += 1

        # Calculate discharged throughput during this action interval
        discharged_pcu = max(0.0, self.sim._total_throughput_pcu - initial_throughput)

        # Calculate reward
        reward, reward_breakdown = self.reward_calculator.calculate(
            prev_approaches=prev_approaches,
            curr_approaches=self.sim.approaches,
            throughput_discharged_pcu=discharged_pcu,
            duration_sec=green_duration,
            phase_switched=phase_switched,
        )
        self.total_episode_reward += reward

        # Build new observation vector
        obs_vector = self.observation_builder.build(
            approaches=self.sim.approaches,
            active_phase_id=self.active_phase_id,
            elapsed_in_phase=self.elapsed_in_phase,
            current_step=self.sim.step_count,
            total_steps=self.max_steps,
        )
        self.last_observation = obs_vector

        # Termination & Truncation
        truncated = bool(
            self.sim.step_count >= self.max_steps
            or self.sim.step_count >= self.sim.total_steps
        )
        terminated = False  # Continuous traffic simulation is time-truncated

        # Info dictionary with comprehensive breakdown
        starvation_count = sum(1 for a in self.sim.approaches.values() if a.is_starving)
        info = {
            "action": int(action),
            "action_desc": self.action_decoder.describe(action),
            "selected_phase_id": target_phase_id,
            "green_duration_sec": green_duration,
            "phase_switched": phase_switched,
            "simulation_step": self.sim.step_count,
            "total_active_queue_pcu": self.last_step_obs.total_active_queue_pcu if self.last_step_obs else 0.0,
            "average_delay_sec": self.last_step_obs.average_delay_sec if self.last_step_obs else 0.0,
            "total_throughput_pcu": self.sim._total_throughput_pcu,
            "starvation_occurrences": starvation_count,
            "simulator_backend": self.sim.backend,
            "reward_breakdown": reward_breakdown,
            "total_episode_reward": round(self.total_episode_reward, 4),
        }

        return obs_vector, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Renders ASCII state summary of the intersection."""
        if self.sim is None:
            return "Environment not initialized."
        q_n = self.sim.approaches.get("North_Approach", ApproachState(approach_id="N", approach_name="N", direction="UP", camera_id="a2", junction_id="j")).estimated_queue_pcu
        q_s = self.sim.approaches.get("South_Approach", ApproachState(approach_id="S", approach_name="S", direction="DOWN", camera_id="a2", junction_id="j")).estimated_queue_pcu
        q_e = self.sim.approaches.get("East_Approach", ApproachState(approach_id="E", approach_name="E", direction="RIGHT", camera_id="a3", junction_id="j")).estimated_queue_pcu
        q_w = self.sim.approaches.get("West_Approach", ApproachState(approach_id="W", approach_name="W", direction="LEFT", camera_id="a3", junction_id="j")).estimated_queue_pcu

        summary = (
            f"[Step {self.sim.step_count}/{self.max_steps}] "
            f"Active: Phase {self.active_phase_id} | "
            f"Queues: N={q_n:.1f}, S={q_s:.1f}, E={q_e:.1f}, W={q_w:.1f} PCU | "
            f"Reward: {self.total_episode_reward:.2f}"
        )
        if self.config.render_mode == "human":
            print(summary)
        return summary

    def close(self):
        """Cleanup environment resources."""
        if self.sim is not None:
            self.sim.close()
        self.sim = None
