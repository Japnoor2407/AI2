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

        self.audit_log: List[SafetyEvent] = []
        self.total_violations_prevented: int = 0

    def reset(self):
        self.audit_log = []
        self.total_violations_prevented = 0

    def validate_and_enforce(
        self,
        decision: ControllerDecision,
        approaches: Dict[str, ApproachState],
        current_step: int,
        timestamp_str: str,
    ) -> Tuple[ControllerDecision, List[SafetyEvent]]:
        """
        Validates the controller decision using the decision's own state fields.
        Returns (approved_decision, list_of_safety_events).
        """
        # Delegate to the internal implementation using decision's own fields
        approved, event = self._validate(
            raw_decision=decision,
            current_active_phase_id=decision.active_phase_id,
            current_signal_state=decision.signal_state,
            elapsed_in_state=decision.elapsed_in_phase,
            approaches=approaches,
        )
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
            override_reasons.append(
                f"Invalid phase_id ({requested_phase_id}) requested; falling back to active phase {current_active_phase_id}."
            )
            approved_phase_id = current_active_phase_id

        # Check for non-finite or negative remaining times
        if (
            math.isnan(approved_decision.time_remaining_in_state)
            or math.isinf(approved_decision.time_remaining_in_state)
            or approved_decision.time_remaining_in_state < 0
        ):
            approved_decision.time_remaining_in_state = 0

        # ── 2. Clearance Interval Safety (No Instantaneous Phase Jumping) ──
        # If currently in YELLOW or ALL-RED, controller cannot spontaneously switch to GREEN of another phase
        if current_signal_state == SignalState.YELLOW:
            active_constraints.append("YELLOW_CLEARANCE_ACTIVE")
            if raw_decision.signal_state == SignalState.GREEN:
                is_overridden = True
                override_reasons.append("Cannot interrupt Yellow clearance with instantaneous Green.")
                approved_decision.signal_state = SignalState.YELLOW
                approved_phase_id = current_active_phase_id

        elif current_signal_state == SignalState.ALL_RED:
            active_constraints.append("ALL_RED_CLEARANCE_ACTIVE")
            if raw_decision.signal_state == SignalState.GREEN and elapsed_in_state < self.all_red_sec:
                is_overridden = True
                override_reasons.append("All-Red junction clearance interval has not completed.")
                approved_decision.signal_state = SignalState.ALL_RED
                approved_phase_id = current_active_phase_id

        # ── 3. Minimum Green Enforcement ──
        if (
            current_signal_state == SignalState.GREEN
            and raw_decision.signal_state in (SignalState.YELLOW, SignalState.RED)
            and elapsed_in_state < self.min_green_sec
        ):
            active_constraints.append("MIN_GREEN_VIOLATION")
            is_overridden = True
            remaining = self.min_green_sec - elapsed_in_state
            override_reasons.append(
                f"Minimum green ({self.min_green_sec}s) not yet satisfied ({remaining}s remaining)."
            )
            approved_decision.signal_state = SignalState.GREEN
            approved_decision.is_transitioning = False
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
                f"Maximum green ({self.max_green_sec}s) exceeded; initiating Yellow clearance."
            )
            approved_decision.signal_state = SignalState.YELLOW
            approved_decision.is_transitioning = True

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
                        f"Phase {approved_phase_id} violates non-conflicting constraint (crosses NS and EW)."
                    )
                    approved_phase_id = 1
                    approved_decision.signal_state = SignalState.ALL_RED

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

