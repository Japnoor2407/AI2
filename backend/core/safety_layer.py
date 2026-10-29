"""
Safety Supervisor Layer (Phase 2 Requirement).

An independent safety boundary that sits between any controller (Fixed,
Actuated, Adaptive, ML) and the final signal actuation. Ensures zero
conflicting greens, strict clearance intervals (Yellow/All-Red), min/max green
bounds, and logs every override event.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from backend.core.schema import (
    ApproachState,
    ControllerDecision,
    ControllerMode,
    PhaseConfig,
    SafetyEvent,
    SignalState,
)
from backend.core.config import (
    MIN_GREEN_SEC,
    MAX_GREEN_SEC,
    YELLOW_CLEARANCE_SEC,
    ALL_RED_CLEARANCE_SEC,
)

logger = logging.getLogger(__name__)


class SafetySupervisor:
    """
    Independent Safety Supervisor.
    Enforces signal safety invariants and audits decisions.
    Maintains autonomous tracking of junction state to detect illegal transitions.
    """

    def __init__(
        self,
        intersection_id: str,
        phases: List[PhaseConfig],
        min_green_sec: int = MIN_GREEN_SEC,
        max_green_sec: int = MAX_GREEN_SEC,
        yellow_sec: int = YELLOW_CLEARANCE_SEC,
        all_red_sec: int = ALL_RED_CLEARANCE_SEC,
    ):
        self.intersection_id = intersection_id
        self.phases = phases
        self.phase_map: Dict[int, PhaseConfig] = {p.phase_id: p for p in phases}
        self.min_green_sec = min_green_sec
        self.max_green_sec = max_green_sec
        self.yellow_sec = yellow_sec
        self.all_red_sec = all_red_sec

        # Autonomous junction state tracking
        self.current_phase_id: int = phases[0].phase_id if phases else 1
        self.current_signal_state: SignalState = SignalState.GREEN
        self.elapsed_in_state: int = 0

        self.audit_log: List[SafetyEvent] = []
        self.total_violations_prevented: int = 0

    def reset(self):
        self.audit_log = []
        self.total_violations_prevented = 0
        self.current_phase_id = self.phases[0].phase_id if self.phases else 1
        self.current_signal_state = SignalState.GREEN
        self.elapsed_in_state = 0

    def validate_and_enforce(
        self,
        decision: ControllerDecision,
        approaches: Dict[str, ApproachState],
        current_step: int,
        timestamp_str: str,
        current_active_phase_id: Optional[int] = None,
        current_signal_state: Optional[SignalState] = None,
        elapsed_in_state: Optional[int] = None,
    ) -> Tuple[ControllerDecision, List[SafetyEvent]]:
        """
        Validates the controller decision using either provided state or internally tracked state.
        Returns (approved_decision, list_of_safety_events).
        """
        curr_phase = current_active_phase_id if current_active_phase_id is not None else self.current_phase_id
        curr_state = current_signal_state if current_signal_state is not None else self.current_signal_state
        curr_elapsed = elapsed_in_state if elapsed_in_state is not None else self.elapsed_in_state

        approved, event = self._validate(
            raw_decision=decision,
            current_active_phase_id=curr_phase,
            current_signal_state=curr_state,
            elapsed_in_state=curr_elapsed,
            approaches=approaches,
        )

        # Update autonomous state tracking
        if approved.active_phase_id == self.current_phase_id and approved.signal_state == self.current_signal_state:
            self.elapsed_in_state += 1
        else:
            self.current_phase_id = approved.active_phase_id
            self.current_signal_state = approved.signal_state
            self.elapsed_in_state = 1

        return approved, [event] if event else []

    def _validate(
        self,
        raw_decision: ControllerDecision,
        current_active_phase_id: int,
        current_signal_state: SignalState,
        elapsed_in_state: int,
        approaches: Dict[str, ApproachState],
    ) -> Tuple[ControllerDecision, Optional[SafetyEvent]]:
        """
        Validates the raw controller decision.
        If any safety rule is violated, overrides the decision and records a SafetyEvent.
        """
        is_overridden = False
        override_reasons: List[str] = []
        active_constraints: List[str] = []

        approved_decision = raw_decision.model_copy()
        requested_phase_id = raw_decision.active_phase_id
        approved_phase_id = requested_phase_id

        # ── 1. Sanity & Numerical Validity Check ──
        if (
            not isinstance(requested_phase_id, int)
            or requested_phase_id not in self.phase_map
        ):
            is_overridden = True
            active_constraints.append("INVALID_PHASE_ID")
            override_reasons.append(
                f"Invalid phase_id ({requested_phase_id}) requested; falling back to active phase {current_active_phase_id}."
            )
            approved_phase_id = current_active_phase_id
            approved_decision.active_phase_id = approved_phase_id

        # Check for non-finite or negative remaining times
        if (
            math.isnan(approved_decision.time_remaining_in_state)
            or math.isinf(approved_decision.time_remaining_in_state)
            or approved_decision.time_remaining_in_state < 0
        ):
            active_constraints.append("INVALID_NUMERICAL_VALUE")
            approved_decision.time_remaining_in_state = 0

        # ── 2. Clearance Interval Safety (No Premature Interruption) ──
        if current_signal_state == SignalState.YELLOW:
            active_constraints.append("YELLOW_CLEARANCE_ACTIVE")
            if raw_decision.signal_state != SignalState.YELLOW:
                if elapsed_in_state < self.yellow_sec:
                    is_overridden = True
                    rem = self.yellow_sec - elapsed_in_state
                    override_reasons.append(
                        f"Cannot interrupt Yellow clearance before {self.yellow_sec}s complete ({rem}s remaining)."
                    )
                    approved_decision.signal_state = SignalState.YELLOW
                    approved_phase_id = current_active_phase_id
                    approved_decision.is_transitioning = True
                    approved_decision.transition_stage = "YELLOW"

        elif current_signal_state == SignalState.ALL_RED:
            active_constraints.append("ALL_RED_CLEARANCE_ACTIVE")
            if raw_decision.signal_state == SignalState.GREEN and elapsed_in_state < self.all_red_sec:
                is_overridden = True
                rem = self.all_red_sec - elapsed_in_state
                override_reasons.append(
                    f"All-Red clearance interval has not completed ({rem}s remaining)."
                )
                approved_decision.signal_state = SignalState.ALL_RED
                approved_phase_id = current_active_phase_id
                approved_decision.is_transitioning = True
                approved_decision.transition_stage = "ALL_RED"

        # ── 3. Minimum Green Enforcement (No Premature Green Termination) ──
        if (
            current_signal_state == SignalState.GREEN
            and raw_decision.signal_state in (SignalState.YELLOW, SignalState.RED, SignalState.ALL_RED)
            and elapsed_in_state < self.min_green_sec
        ):
            active_constraints.append("MIN_GREEN_VIOLATION")
            is_overridden = True
            remaining = self.min_green_sec - elapsed_in_state
            override_reasons.append(
                f"Minimum green ({self.min_green_sec}s) not yet satisfied ({remaining}s remaining). Premature termination blocked."
            )
            approved_decision.signal_state = SignalState.GREEN
            approved_decision.is_transitioning = False
            approved_decision.transition_stage = None
            approved_phase_id = current_active_phase_id

        # ── 4. Maximum Green Enforcement ──
        if (
            current_signal_state == SignalState.GREEN
            and elapsed_in_state >= self.max_green_sec
            and raw_decision.signal_state == SignalState.GREEN
        ):
            active_constraints.append("MAX_GREEN_EXCEEDED")
            is_overridden = True
            override_reasons.append(
                f"Maximum green ({self.max_green_sec}s) exceeded; initiating mandatory Yellow clearance."
            )
            approved_decision.signal_state = SignalState.YELLOW
            approved_decision.is_transitioning = True
            approved_decision.transition_stage = "YELLOW"

        # ── 5. Non-Conflicting Phase Invariant Check ──
        if approved_decision.signal_state == SignalState.GREEN:
            active_constraints.append("CONFLICT_FREE_INVARIANT")
            candidate_phase = self.phase_map.get(approved_phase_id)
            if candidate_phase:
                served = set(candidate_phase.served_approaches)
                ns = {"North_Approach", "South_Approach"}
                ew = {"East_Approach", "West_Approach"}
                # Conflict if phase attempts to serve crossing movements together
                if (served & ns) and (served & ew):
                    is_overridden = True
                    override_reasons.append(
                        f"Phase {approved_phase_id} violates non-conflicting constraint (crosses NS and EW simultaneously)."
                    )
                    approved_phase_id = current_active_phase_id
                    approved_decision.signal_state = SignalState.ALL_RED
                    approved_decision.is_transitioning = True
                    approved_decision.transition_stage = "ALL_RED"

        # ── 6. Missing Sensor / Approach Data Handling ──
        if not approaches:
            active_constraints.append("MISSING_SENSOR_DATA")

        # ── Apply Overrides ──
        safety_event = None
        if is_overridden:
            self.total_violations_prevented += 1
            approved_decision.safety_override = True
            reason_text = " | ".join(override_reasons)
            approved_decision.safety_override_reason = reason_text
            approved_decision.active_phase_id = approved_phase_id
            if approved_phase_id in self.phase_map:
                approved_decision.phase_name = self.phase_map[approved_phase_id].name

            safety_event = SafetyEvent(
                step=raw_decision.step,
                timestamp_str=raw_decision.timestamp_str,
                intersection_id=self.intersection_id,
                previous_phase_id=current_active_phase_id,
                requested_phase_id=requested_phase_id,
                approved_phase_id=approved_phase_id,
                is_overridden=True,
                override_reason=reason_text,
                active_constraints=active_constraints,
            )
            self.audit_log.append(safety_event)
            logger.warning(
                f"[SAFETY OVERRIDE step={raw_decision.step}] {reason_text} "
                f"Req={requested_phase_id} -> App={approved_phase_id}"
            )

        return approved_decision, safety_event

    def safe_fallback_decision(
        self,
        current_step: int,
        timestamp_str: str,
        exception_reason: str = "Controller failure",
    ) -> Tuple[ControllerDecision, SafetyEvent]:
        """
        Failsafe mechanism: generates a safe clearance decision when a controller raises an unhandled exception.
        """
        if self.current_signal_state == SignalState.GREEN:
            next_state = SignalState.YELLOW
            duration = self.yellow_sec
        else:
            next_state = SignalState.ALL_RED
            duration = self.all_red_sec

        decision = ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=ControllerMode.ADAPTIVE,
            active_phase_id=self.current_phase_id,
            phase_name=self.phase_map.get(
                self.current_phase_id,
                PhaseConfig(phase_id=self.current_phase_id, name="Safety Clearance", served_approaches=[])
            ).name,
            signal_state=next_state,
            time_remaining_in_state=duration,
            elapsed_in_phase=1,
            safety_override=True,
            safety_override_reason=f"Failsafe triggered: {exception_reason}",
            decision_rationale=f"FAILSAFE CLEARANCE: {exception_reason}",
            is_transitioning=True,
            transition_stage=next_state.value,
        )

        self.total_violations_prevented += 1
        event = SafetyEvent(
            step=current_step,
            timestamp_str=timestamp_str,
            intersection_id=self.intersection_id,
            previous_phase_id=self.current_phase_id,
            requested_phase_id=self.current_phase_id,
            approved_phase_id=self.current_phase_id,
            is_overridden=True,
            override_reason=f"Failsafe triggered: {exception_reason}",
            active_constraints=["CONTROLLER_EXCEPTION_FAILSAFE"],
        )
        self.audit_log.append(event)
        self.current_signal_state = next_state
        self.elapsed_in_state = 1
        return decision, event
