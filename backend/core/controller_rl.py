"""
RLTrafficController — Deep Reinforcement Learning (PPO) traffic signal controller.

Conforms to the same step() / reset() interface as:
  - FixedTimeController  (backend/core/controller_baseline.py)
  - AdaptiveController   (backend/core/controller_adaptive.py)

The controller respects the training contract: the PPO model selects a phase
and green duration ONCE, then holds that decision for the full allocated
green window before querying the model again. This matches how the model was
trained (one action per interval, not one action per second).

Supports both 2-phase dual-arterial layouts and 4-phase protected single-approach
layouts via intelligent demand-weighted routing and starvation preemption.

If no model is loaded, falls back gracefully to AdaptiveController.
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
    STARVATION_THRESHOLD_SEC,
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

    KEY DESIGN:
    1. The PPO model is queried at the START of each green interval.
       It selects an arterial movement and green duration; the controller HOLDS
       that decision for the full duration before querying again.
    2. Maps arterial decisions (North-South vs East-West) to actual layout phases
       (e.g. 4-phase protected movement) by giving green to the approach with
       highest queue and delay demand.
    3. Guarantees starvation safety preemption for neglected approaches.
    """

    def __init__(
        self,
        phases: List[PhaseConfig],
        episode_duration_sec: int = 1800,
        fallback_controller=None,
    ):
        self.phases = phases or []
        self.episode_duration_sec = episode_duration_sec

        # State tracking — current hold window
        self.active_phase_id: int = phases[0].phase_id if phases else 1
        self.elapsed_in_phase: int = 0
        self._allocated_green: int = MIN_GREEN_SEC
        self._green_countdown: int = 0   # seconds remaining in current green hold
        self._in_transition: bool = False
        self._transition_remaining: int = 0
        self._transition_stage: str = "YELLOW"  # "YELLOW" or "ALL_RED"
        self._pending_phase_id: Optional[int] = None
        self._active_rationale: str = "PPO hold interval"

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

    def _get_phase_name(self, phase_id: int) -> str:
        for p in self.phases:
            if p.phase_id == phase_id:
                return p.name
        return f"Phase {phase_id}"

    def _map_arterial_to_phase(
        self, arterial_phase_id: int, approaches: Dict[str, ApproachState]
    ) -> Tuple[int, str]:
        """
        Maps a 2-phase arterial policy decision (NS=1, EW=2) to the active junction phase layout.
        If the layout is 4-phase (North=1, East=2, South=3, West=4):
          - NS arterial (1) routes to whichever of North (1) or South (3) has greater queue & wait demand.
          - EW arterial (2) routes to whichever of East (2) or West (4) has greater queue & wait demand.
        Also enforces starvation preemption if any approach exceeds STARVATION_THRESHOLD_SEC.
        """
        # 1. Starvation preemption safety (proactive threshold at 65s before 80s hard limit)
        for app_id, app in approaches.items():
            if app.time_since_last_served_sec >= 65.0:
                for p in self.phases:
                    if app_id in p.served_approaches:
                        return p.phase_id, f"Starvation preemption: {p.name} ({app.time_since_last_served_sec:.0f}s wait)"

        # 2. 4-phase mapping
        if len(self.phases) == 4:
            if arterial_phase_id == 1:
                # North (1) vs South (3)
                qn = getattr(approaches.get("North_Approach"), "estimated_queue_pcu", 0.0)
                qs = getattr(approaches.get("South_Approach"), "estimated_queue_pcu", 0.0)
                wn = getattr(approaches.get("North_Approach"), "time_since_last_served_sec", 0.0)
                ws = getattr(approaches.get("South_Approach"), "time_since_last_served_sec", 0.0)
                score_n = qn * 1.5 + (wn / 15.0)
                score_s = qs * 1.5 + (ws / 15.0)
                target = 3 if score_s > score_n else 1
                name = self._get_phase_name(target)
                return target, f"PPO NS policy → {name} (Q: {max(qn, qs):.1f} PCU)"
            else:
                # East (2) vs West (4)
                qe = getattr(approaches.get("East_Approach"), "estimated_queue_pcu", 0.0)
                qw = getattr(approaches.get("West_Approach"), "estimated_queue_pcu", 0.0)
                we = getattr(approaches.get("East_Approach"), "time_since_last_served_sec", 0.0)
                ww = getattr(approaches.get("West_Approach"), "time_since_last_served_sec", 0.0)
                score_e = qe * 1.5 + (we / 15.0)
                score_w = qw * 1.5 + (ww / 15.0)
                target = 4 if score_w > score_e else 2
                name = self._get_phase_name(target)
                return target, f"PPO EW policy → {name} (Q: {max(qe, qw):.1f} PCU)"

        # 3. Direct match for 2-phase layout
        valid_ids = [p.phase_id for p in self.phases]
        if arterial_phase_id in valid_ids:
            return arterial_phase_id, f"PPO selected: {self._get_phase_name(arterial_phase_id)}"
        default_id = valid_ids[0] if valid_ids else 1
        return default_id, f"PPO default: {self._get_phase_name(default_id)}"

    # ──────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset controller state for a new episode."""
        self.active_phase_id = self.phases[0].phase_id if self.phases else 1
        self.elapsed_in_phase = 0
        self._allocated_green = MIN_GREEN_SEC
        self._green_countdown = 0
        self._in_transition = False
        self._transition_remaining = 0
        self._pending_phase_id = None
        self._active_rationale = "PPO hold interval"
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
        """
        # ── Fallback path (model not loaded) ───────────────────
        if not rl_inference.is_ready():
            if self._fallback is not None:
                decision = self._fallback.step(current_step, timestamp_str, approaches)
                decision.mode = ControllerMode.DEEP_RL
                decision.decision_rationale = (
                    "[RL Fallback] PPO model not trained yet — "
                    + decision.decision_rationale
                )
                return decision
            return self._default_decision(current_step, timestamp_str)

        # ── Transition (yellow / all-red) ─────────────────────
        if self._in_transition:
            return self._handle_transition(current_step, timestamp_str)

        # ── Green hold: re-emit current decision or gap-out if queue fully cleared ──
        if self._green_countdown > 0:
            active_p = next((p for p in self.phases if p.phase_id == self.active_phase_id), None)
            served_apps = active_p.served_approaches if active_p else []
            active_queue = sum(getattr(approaches.get(a), "estimated_queue_pcu", 0.0) for a in served_apps)
            
            # Check if active approach queue has discharged after safety min_green (10s)
            # while other waiting approaches have pending queues or accumulating delay
            waiting_apps = [a for k, a in approaches.items() if k not in served_apps]
            max_waiting_queue = max([getattr(a, "estimated_queue_pcu", 0.0) for a in waiting_apps], default=0.0)
            max_waiting_time = max([getattr(a, "time_since_last_served_sec", 0.0) for a in waiting_apps], default=0.0)
            
            can_gap_out = (
                self.elapsed_in_phase >= MIN_GREEN_SEC
                and active_queue < 0.2
                and (max_waiting_queue >= 0.8 or max_waiting_time >= 25.0)
            )
            
            if can_gap_out:
                # Early gap-out to pending demand: avoid wasting green capacity on empty approaches
                self._green_countdown = 0
            else:
                self._green_countdown -= 1
                self.elapsed_in_phase += 1
                phase_name = self._get_phase_name(self.active_phase_id)
                decision = ControllerDecision(
                    step=current_step,
                    timestamp_str=timestamp_str,
                    mode=ControllerMode.DEEP_RL,
                    active_phase_id=self.active_phase_id,
                    phase_name=phase_name,
                    signal_state=SignalState.GREEN,
                    time_remaining_in_state=self._green_countdown,
                    elapsed_in_phase=self.elapsed_in_phase,
                    decision_rationale=f"{self._active_rationale} ({self._green_countdown}s remaining)",
                )
                return decision

        # ── Green expired: query model for next action ────────
        # For observation vector, arterial phase 1 is active if current phase is on NS corridor
        active_arterial = 1 if (self.active_phase_id in (1, 3)) else 2
        try:
            obs = self._observation_builder.build(
                approaches=approaches,
                active_phase_id=active_arterial,
                elapsed_in_phase=self.elapsed_in_phase,
                current_step=current_step,
                total_steps=self.episode_duration_sec,
            )
            art_phase_id, green_duration, desc = rl_inference.predict(obs, deterministic=True)
        except Exception as exc:
            logger.error(f"RLTrafficController: inference error at step {current_step}: {exc}")
            if self._fallback is not None:
                decision = self._fallback.step(current_step, timestamp_str, approaches)
                decision.mode = ControllerMode.DEEP_RL
                decision.decision_rationale = f"[RL Error Fallback] {exc}"
                return decision
            return self._default_decision(current_step, timestamp_str)

        green_duration = int(np.clip(green_duration, MIN_GREEN_SEC, MAX_GREEN_SEC))
        target_phase_id, rationale = self._map_arterial_to_phase(art_phase_id, approaches)
        phase_switched = (target_phase_id != self.active_phase_id)

        # ── Phase switch: start transition sequence ───────────
        if phase_switched:
            self._pending_phase_id = target_phase_id
            self._allocated_green = green_duration
            self._active_rationale = rationale
            self._in_transition = True
            self._transition_stage = "YELLOW"
            self._transition_remaining = YELLOW_CLEARANCE_SEC
            return self._handle_transition(current_step, timestamp_str)

        # ── Same phase: apply new green duration ──────────────
        self._allocated_green = green_duration
        self._green_countdown = green_duration - 1  # this second counts
        self.elapsed_in_phase += 1
        self._active_rationale = rationale

        phase_name = self._get_phase_name(self.active_phase_id)
        decision = ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=ControllerMode.DEEP_RL,
            active_phase_id=self.active_phase_id,
            phase_name=phase_name,
            signal_state=SignalState.GREEN,
            time_remaining_in_state=green_duration,
            elapsed_in_phase=self.elapsed_in_phase,
            decision_rationale=f"{rationale} (held for {green_duration}s)",
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
                phase_name=f"{self._get_phase_name(self.active_phase_id)} Yellow Clearance",
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
        phase_id = self.phases[0].phase_id if self.phases else 1
        return ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=ControllerMode.DEEP_RL,
            active_phase_id=phase_id,
            phase_name=self._get_phase_name(phase_id),
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
