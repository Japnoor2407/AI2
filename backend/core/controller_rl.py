"""
RLTrafficController — Deep Reinforcement Learning (PPO) traffic signal controller.

Conforms to the same step() / reset() interface as:
  - FixedTimeController  (backend/core/controller_baseline.py)
  - AdaptiveController   (backend/core/controller_adaptive.py)

The controller respects the training contract: the PPO model selects a phase
and green duration ONCE, then holds that decision for the full allocated
green window before querying the model again. This matches how the model was
trained (one action per interval, not one action per second).

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

    KEY DESIGN: The PPO model is queried at the START of each green interval.
    It selects a (phase, duration) pair and the controller HOLDS that decision
    for the full duration before querying again. This matches the training
    contract where each env.step() covers one complete green interval.

    Without this hold mechanism, the model would be called every second
    and always emit 10s minimums, causing phase thrash and blocking throughput.
    """

    def __init__(
        self,
        phases: List[PhaseConfig],
        episode_duration_sec: int = 1800,
        fallback_controller=None,
    ):
        self.phases = phases
        self.episode_duration_sec = episode_duration_sec

        # State tracking — current hold window
        self.active_phase_id: int = 1
        self.elapsed_in_phase: int = 0
        self._allocated_green: int = MIN_GREEN_SEC
        self._green_countdown: int = 0   # seconds remaining in current green hold
        self._in_transition: bool = False
        self._transition_remaining: int = 0
        self._transition_stage: str = "YELLOW"  # "YELLOW" or "ALL_RED"
        self._pending_phase_id: Optional[int] = None

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
        self._allocated_green = MIN_GREEN_SEC
        self._green_countdown = 0
        self._in_transition = False
        self._transition_remaining = 0
        self._pending_phase_id = None
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

        Uses a hold mechanism: the PPO model is only queried when the current
        green allocation has expired. During the hold window the last decision
        is re-emitted unchanged (as every other controller does during green).

        Args:
            current_step:  Simulation step index.
            timestamp_str: Wall-clock time string (e.g. "09:15:32").
            approaches:    Current approach states from JunctionSimulator.

        Returns:
            ControllerDecision for this second.
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
            return self._default_decision(current_step, timestamp_str)

        # ── Transition (yellow / all-red) ─────────────────────
        if self._in_transition:
            return self._handle_transition(current_step, timestamp_str)

        # ── Green hold: re-emit current decision without re-querying model ──
        if self._green_countdown > 0:
            self._green_countdown -= 1
            self.elapsed_in_phase += 1
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
                time_remaining_in_state=self._green_countdown,
                elapsed_in_phase=self.elapsed_in_phase,
                decision_rationale=f"PPO hold: {phase_name} ({self._green_countdown}s remaining)",
            )
            return decision

        # ── Green expired: query model for next action ────────
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
            if self._fallback is not None:
                decision = self._fallback.step(current_step, timestamp_str, approaches)
                decision.mode = ControllerMode.DEEP_RL
                decision.decision_rationale = f"[RL Error Fallback] {exc}"
                return decision
            return self._default_decision(current_step, timestamp_str)

        green_duration = int(np.clip(green_duration, MIN_GREEN_SEC, MAX_GREEN_SEC))
        phase_switched = (phase_id != self.active_phase_id)

        # ── Phase switch: start transition sequence ───────────
        if phase_switched:
            self._pending_phase_id = phase_id
            self._allocated_green = green_duration
            self._in_transition = True
            self._transition_stage = "YELLOW"
            self._transition_remaining = YELLOW_CLEARANCE_SEC
            return self._handle_transition(current_step, timestamp_str)

        # ── Same phase: apply new green duration ──────────────
        self._allocated_green = green_duration
        self._green_countdown = green_duration - 1  # this second counts
        self.elapsed_in_phase += 1

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

    def _handle_transition(self, current_step: int, timestamp_str: str) -> ControllerDecision:
        """Emit yellow or all-red clearance seconds during a phase switch."""
        if self._transition_stage == "YELLOW":
            self._transition_remaining -= 1
            decision = ControllerDecision(
                step=current_step,
                timestamp_str=timestamp_str,
                mode=ControllerMode.DEEP_RL,
                active_phase_id=self.active_phase_id,
                phase_name=f"Phase {self.active_phase_id} Yellow Clearance",
                signal_state=SignalState.YELLOW,
                time_remaining_in_state=max(0, self._transition_remaining),
                elapsed_in_phase=self.elapsed_in_phase,
                is_transitioning=True,
                transition_stage="YELLOW",
                decision_rationale="PPO yellow clearance interval",
            )
            if self._transition_remaining <= 0:
                self._transition_stage = "ALL_RED"
                self._transition_remaining = ALL_RED_CLEARANCE_SEC
        else:  # ALL_RED
            self._transition_remaining -= 1
            decision = ControllerDecision(
                step=current_step,
                timestamp_str=timestamp_str,
                mode=ControllerMode.DEEP_RL,
                active_phase_id=self.active_phase_id,
                phase_name="All-Red Clearance",
                signal_state=SignalState.ALL_RED,
                time_remaining_in_state=max(0, self._transition_remaining),
                elapsed_in_phase=self.elapsed_in_phase,
                is_transitioning=True,
                transition_stage="ALL_RED",
                decision_rationale="PPO inter-green safety clearance",
            )
            if self._transition_remaining <= 0:
                # Transition complete — switch to new phase
                self._in_transition = False
                if self._pending_phase_id is not None:
                    self.active_phase_id = self._pending_phase_id
                    self._pending_phase_id = None
                self.elapsed_in_phase = 0
                self._green_countdown = self._allocated_green
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
