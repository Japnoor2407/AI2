"""
RLTrafficController — Deep Reinforcement Learning (PPO) traffic signal controller.

Conforms to the same step() / reset() interface as:
  - FixedTimeController  (backend/core/controller_baseline.py)
  - AdaptiveController   (backend/core/controller_adaptive.py)

This allows AppState to swap it in without changes to the simulation loop.

The controller:
  1. Builds the 21-dim observation vector from current approach states.
  2. Calls the loaded PPO model for a prediction.
  3. Passes the decoded decision through RLSafetyValidator.
  4. Returns the approved ControllerDecision.

If no model is loaded, falls back to AdaptiveController.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.core.schema import (
    ApproachState,
    ControllerDecision,
    ControllerMode,
    PhaseConfig,
    SignalState,
)
from backend.core.config import (
    MIN_GREEN_SEC,
    MAX_GREEN_SEC,
    YELLOW_CLEARANCE_SEC,
    ALL_RED_CLEARANCE_SEC,
)
from backend.rl.inference import rl_inference
from backend.rl.observation import ObservationBuilder
from backend.rl.action_space import ActionDecoder
from backend.rl.safety import RLSafetyValidator

logger = logging.getLogger(__name__)


class RLTrafficController:
    """
    PPO-based deep RL traffic signal controller.

    Matches the step(current_step, timestamp_str, approaches) → ControllerDecision
    interface used by all other controllers and the AppState simulation loop.
    """

    def __init__(
        self,
        phases: List[PhaseConfig],
        episode_duration_sec: int = 1800,
        fallback_controller=None,
    ):
        self.phases = phases
        self.episode_duration_sec = episode_duration_sec

        # State tracking
        self.active_phase_id: int = 1
        self.elapsed_in_phase: int = 0
        self.current_signal_state: SignalState = SignalState.GREEN
        self._time_in_current_state: int = 0
        self._allocated_green: int = MIN_GREEN_SEC

        # Helpers
        self._observation_builder = ObservationBuilder()
        self._action_decoder = ActionDecoder()
        self._safety_validator: Optional[RLSafetyValidator] = None

        # Lazy-init safety validator (needs phase list)
        if phases:
            self._init_safety(phases)

        # Fallback when model not loaded
        self._fallback = fallback_controller

        # Attempt to load model (non-blocking — fails gracefully)
        if not rl_inference.is_ready():
            loaded = rl_inference.load()
            if loaded:
                logger.info("RLTrafficController: PPO model loaded.")
            else:
                logger.warning(
                    "RLTrafficController: no trained model found — "
                    "falling back to AdaptiveController."
                )

    def _init_safety(self, phases: List[PhaseConfig]):
        if phases:
            self._safety_validator = RLSafetyValidator(
                intersection_id="AlankarChowk",
                phases=phases,
            )

    # ──────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset controller state for a new episode."""
        self.active_phase_id = 1
        self.elapsed_in_phase = 0
        self.current_signal_state = SignalState.GREEN
        self._time_in_current_state = 0
        self._allocated_green = MIN_GREEN_SEC
        if self._safety_validator:
            self._safety_validator.reset()
        if self._fallback:
            self._fallback.reset()

    def step(
        self,
        current_step: int,
        timestamp_str: str,
        approaches: Dict[str, ApproachState],
    ) -> ControllerDecision:
        """
        Produce a ControllerDecision for the current simulation step.

        Args:
            current_step:  Simulation step index.
            timestamp_str: Wall-clock time string (e.g. "09:15:32").
            approaches:    Current approach states from JunctionSimulator.

        Returns:
            Safety-validated ControllerDecision.
        """
        # ── Fallback path (model not loaded) ───────────────────
        if not rl_inference.is_ready():
            if self._fallback is not None:
                decision = self._fallback.step(current_step, timestamp_str, approaches)
                decision.mode = ControllerMode.DEEP_RL  # tag as RL for UI
                decision.decision_rationale = (
                    "[RL Fallback] PPO model not trained yet — "
                    + decision.decision_rationale
                )
                return decision
            # No fallback — produce a safe default
            return self._default_decision(current_step, timestamp_str)

        # ── RL inference path ────────────────────────────────
        try:
            obs = self._observation_builder.build(
                approaches=approaches,
                active_phase_id=self.active_phase_id,
                elapsed_in_phase=self.elapsed_in_phase,
                current_step=current_step,
                total_steps=self.episode_duration_sec,
            )
            phase_id, green_duration, desc = rl_inference.predict(obs, deterministic=True)
        except Exception as exc:
            logger.error(f"RLTrafficController: inference error at step {current_step}: {exc}")
            if self._safety_validator:
                fallback_decision, event = self._safety_validator.get_fallback_decision(
                    current_step=current_step,
                    timestamp_str=timestamp_str,
                    reason=f"RL inference failure: {exc}",
                )
                return fallback_decision
            return self._default_decision(current_step, timestamp_str)

        # Track phase state
        phase_switched = (phase_id != self.active_phase_id)
        if phase_switched:
            self.active_phase_id = phase_id
            self.elapsed_in_phase = 0
        else:
            self.elapsed_in_phase += 1

        green_duration = int(np.clip(green_duration, MIN_GREEN_SEC, MAX_GREEN_SEC))
        phase_name = (
            "Phase 1: North-South Arterial"
            if self.active_phase_id == 1
            else "Phase 2: East-West Crossway"
        )

        decision = ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=ControllerMode.DEEP_RL,
            active_phase_id=self.active_phase_id,
            phase_name=phase_name,
            signal_state=SignalState.GREEN,
            time_remaining_in_state=green_duration,
            elapsed_in_phase=self.elapsed_in_phase,
            decision_rationale=f"PPO selected: {desc}",
        )

        # Safety validation
        if self._safety_validator:
            decision, _ = self._safety_validator.validate(
                decision=decision,
                approaches=approaches,
                current_step=current_step,
                timestamp_str=timestamp_str,
            )

        return decision

    def _default_decision(self, current_step: int, timestamp_str: str) -> ControllerDecision:
        """Minimal safe default decision (Phase 1 green, minimum duration)."""
        return ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=ControllerMode.DEEP_RL,
            active_phase_id=1,
            phase_name="Phase 1: North-South Arterial",
            signal_state=SignalState.GREEN,
            time_remaining_in_state=MIN_GREEN_SEC,
            elapsed_in_phase=0,
            decision_rationale="RL safe default (model not ready)",
        )

    @property
    def safety_events(self):
        if self._safety_validator:
            return self._safety_validator.safety_events
        return []

