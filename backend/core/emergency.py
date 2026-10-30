"""
Emergency Vehicle Priority Management (EVP) System.

Enables emergency vehicles (ambulances, fire engines, police response)
to request and receive preemptive green signal priority at intersections
without violating safety invariants or causing conflicting green signals.

Operating Principles:
1. Detection: Detects an emergency vehicle approaching on a specific approach.
2. Conflict-Free Preemption: Identifies the non-conflicting phase that serves the vehicle.
3. Safe Transition: If a different phase is active, safely enforces yellow (3s) and all-red (2s)
   clearance intervals before turning the emergency phase green.
4. Green Extension: If the phase is already green, holds green until the emergency vehicle clears.
5. Audit Logging: Logs every EVP event with timestamp, vehicle type, and clearance latency.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

from backend.core.schema import (
    ApproachState,
    ControllerDecision,
    PhaseConfig,
    SignalState,
)
from backend.core.config import (
    YELLOW_CLEARANCE_SEC,
    ALL_RED_CLEARANCE_SEC,
    MIN_GREEN_SEC,
)

logger = logging.getLogger(__name__)


class EmergencyVehicleType(str, Enum):
    AMBULANCE = "AMBULANCE"
    FIRE_TRUCK = "FIRE_TRUCK"
    POLICE = "POLICE"


class EmergencyPriorityStatus(str, Enum):
    IDLE = "IDLE"
    REQUESTED = "REQUESTED"
    TRANSITIONING_YELLOW = "TRANSITIONING_YELLOW"
    TRANSITIONING_ALL_RED = "TRANSITIONING_ALL_RED"
    ACTIVE_GREEN = "ACTIVE_GREEN"
    CLEARED = "CLEARED"


class EmergencyEvent(BaseModel):
    event_id: str
    vehicle_type: EmergencyVehicleType
    approach_id: str
    target_phase_id: int
    requested_at_step: int
    cleared_at_step: Optional[int] = None
    clearance_latency_sec: Optional[float] = None
    status: EmergencyPriorityStatus = EmergencyPriorityStatus.REQUESTED
    notes: str = ""


class EmergencyPriorityManager:
    """
    Coordinates safe emergency vehicle preemption at the intersection.
    """

    def __init__(
        self,
        phases: List[PhaseConfig],
        yellow_sec: int = YELLOW_CLEARANCE_SEC,
        all_red_sec: int = ALL_RED_CLEARANCE_SEC,
    ):
        self.phases = phases
        self.yellow_sec = yellow_sec
        self.all_red_sec = all_red_sec

        self.active_event: Optional[EmergencyEvent] = None
        self.event_history: List[EmergencyEvent] = []
        self._elapsed_in_stage: int = 0
        self._target_phase: Optional[PhaseConfig] = None

    def reset(self):
        self.active_event = None
        self.event_history.clear()
        self._elapsed_in_stage = 0
        self._target_phase = None

    def request_priority(
        self,
        vehicle_type: EmergencyVehicleType,
        approach_id: str,
        current_step: int,
    ) -> EmergencyEvent:
        """
        Dispatches an emergency vehicle priority request.
        """
        # Find which phase serves this approach
        target_phase = None
        for p in self.phases:
            if approach_id in p.served_approaches:
                target_phase = p
                break

        if target_phase is None:
            target_phase = self.phases[0]

        event_id = f"evp_{current_step}_{vehicle_type.value.lower()}"
        event = EmergencyEvent(
            event_id=event_id,
            vehicle_type=vehicle_type,
            approach_id=approach_id,
            target_phase_id=target_phase.phase_id,
            requested_at_step=current_step,
            status=EmergencyPriorityStatus.REQUESTED,
            notes=f"Emergency preemption requested for {vehicle_type.value} on {approach_id}",
        )

        self.active_event = event
        self._target_phase = target_phase
        self._elapsed_in_stage = 0
        logger.info(f"[EVP] Emergency vehicle priority dispatched: {event.notes}")
        return event

    def clear_active_priority(self, current_step: int):
        """Signals that the emergency vehicle has successfully cleared the intersection."""
        if self.active_event:
            self.active_event.cleared_at_step = current_step
            self.active_event.clearance_latency_sec = float(
                current_step - self.active_event.requested_at_step
            )
            self.active_event.status = EmergencyPriorityStatus.CLEARED
            self.event_history.append(self.active_event)
            logger.info(
                f"[EVP] Emergency vehicle {self.active_event.vehicle_type.value} "
                f"cleared intersection in {self.active_event.clearance_latency_sec}s."
            )
            self.active_event = None
            self._target_phase = None
            self._elapsed_in_stage = 0

    def intercept_decision(
        self,
        normal_decision: ControllerDecision,
        current_step: int,
        timestamp_str: str,
    ) -> ControllerDecision:
        """
        Evaluates active emergency priority and safely overrides normal controller decisions.
        Guarantees clearance intervals (yellow -> all-red -> green).
        """
        if self.active_event is None or self._target_phase is None:
            return normal_decision

        self._elapsed_in_stage += 1
        ev = self.active_event
        target_p = self._target_phase

        # Case 1: Target phase is already active and showing GREEN
        if normal_decision.active_phase_id == target_p.phase_id and normal_decision.signal_state == SignalState.GREEN:
            ev.status = EmergencyPriorityStatus.ACTIVE_GREEN
            return ControllerDecision(
                step=current_step,
                timestamp_str=timestamp_str,
                mode=normal_decision.mode,
                active_phase_id=target_p.phase_id,
                phase_name=target_p.name,
                signal_state=SignalState.GREEN,
                time_remaining_in_state=30,  # hold green for EV clearance
                elapsed_in_phase=normal_decision.elapsed_in_phase,
                decision_rationale=f"EMERGENCY PRIORITY: Green held for {ev.vehicle_type.value} on {ev.approach_id}",
                is_transitioning=False,
                safety_override=True,
                safety_override_reason=f"Emergency Preemption ({ev.vehicle_type.value})",
            )

        # Case 2: Different phase is active -> must transition safely through Yellow and All-Red
        if ev.status == EmergencyPriorityStatus.REQUESTED:
            ev.status = EmergencyPriorityStatus.TRANSITIONING_YELLOW
            self._elapsed_in_stage = 1

        if ev.status == EmergencyPriorityStatus.TRANSITIONING_YELLOW:
            if self._elapsed_in_stage > self.yellow_sec:
                ev.status = EmergencyPriorityStatus.TRANSITIONING_ALL_RED
                self._elapsed_in_stage = 1
                return ControllerDecision(
                    step=current_step,
                    timestamp_str=timestamp_str,
                    mode=normal_decision.mode,
                    active_phase_id=normal_decision.active_phase_id,
                    phase_name="All-Red Clearance",
                    signal_state=SignalState.ALL_RED,
                    time_remaining_in_state=self.all_red_sec,
                    elapsed_in_phase=self._elapsed_in_stage,
                    decision_rationale=f"Emergency preemption clearance for {ev.vehicle_type.value}",
                    is_transitioning=True,
                    transition_stage="ALL_RED",
                    safety_override=True,
                )
            # In yellow
            return ControllerDecision(
                step=current_step,
                timestamp_str=timestamp_str,
                mode=normal_decision.mode,
                active_phase_id=normal_decision.active_phase_id,
                phase_name=normal_decision.phase_name,
                signal_state=SignalState.YELLOW,
                time_remaining_in_state=max(0, self.yellow_sec - self._elapsed_in_stage),
                elapsed_in_phase=self._elapsed_in_stage,
                decision_rationale=f"Emergency yellow transition for {ev.vehicle_type.value}",
                is_transitioning=True,
                transition_stage="YELLOW",
                safety_override=True,
            )

        if ev.status == EmergencyPriorityStatus.TRANSITIONING_ALL_RED:
            if self._elapsed_in_stage > self.all_red_sec:
                ev.status = EmergencyPriorityStatus.ACTIVE_GREEN
                self._elapsed_in_stage = 1
                return ControllerDecision(
                    step=current_step,
                    timestamp_str=timestamp_str,
                    mode=normal_decision.mode,
                    active_phase_id=target_p.phase_id,
                    phase_name=target_p.name,
                    signal_state=SignalState.GREEN,
                    time_remaining_in_state=MIN_GREEN_SEC,
                    elapsed_in_phase=0,
                    decision_rationale=f"EMERGENCY GREEN ACTIVE: Serving {ev.vehicle_type.value} on {ev.approach_id}",
                    is_transitioning=False,
                    safety_override=True,
                )
            # In all-red
            return ControllerDecision(
                step=current_step,
                timestamp_str=timestamp_str,
                mode=normal_decision.mode,
                active_phase_id=normal_decision.active_phase_id,
                phase_name="All-Red Clearance",
                signal_state=SignalState.ALL_RED,
                time_remaining_in_state=max(0, self.all_red_sec - self._elapsed_in_stage),
                elapsed_in_phase=self._elapsed_in_stage,
                decision_rationale=f"Emergency all-red clearance for {ev.vehicle_type.value}",
                is_transitioning=True,
                transition_stage="ALL_RED",
                safety_override=True,
            )

        # Active Green
        return ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=normal_decision.mode,
            active_phase_id=target_p.phase_id,
            phase_name=target_p.name,
            signal_state=SignalState.GREEN,
            time_remaining_in_state=25,
            elapsed_in_phase=self._elapsed_in_stage,
            decision_rationale=f"EMERGENCY GREEN: {ev.vehicle_type.value} proceeding through {target_p.name}",
            is_transitioning=False,
            safety_override=True,
        )
