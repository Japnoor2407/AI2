"""
Max-Pressure Signal Controller (Varaiya, 2013).

A throughput-optimal network traffic control algorithm that dynamically selects
traffic signal phases based on the differential pressure between incoming queues
and downstream exit capacities.

In an isolated 4-way intersection with open exit links, the pressure of a phase
is the sum of queues waiting on its served incoming approaches. The controller
actively chooses the phase with the maximal queue pressure while honoring:
1. Minimum green (10s) to prevent rapid signal flickering.
2. Maximum green (55s) max-out bounds.
3. Yellow clearance (3s) and all-red clearance (2s) safe transitions.
4. Anti-starvation priority overrides (80s threshold).
"""

import logging
from typing import Dict, List, Optional
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

logger = logging.getLogger(__name__)


class MaxPressureController:
    """
    Max-Pressure Traffic Signal Controller.
    Selects the phase with the highest incoming queue pressure while guaranteeing
    safe yellow and all-red transition clearance intervals.
    """

    def __init__(
        self,
        phases: List[PhaseConfig],
        min_green_sec: int = MIN_GREEN_SEC,
        max_green_sec: int = MAX_GREEN_SEC,
        yellow_sec: int = YELLOW_CLEARANCE_SEC,
        all_red_sec: int = ALL_RED_CLEARANCE_SEC,
        starvation_threshold_sec: float = STARVATION_THRESHOLD_SEC,
        pressure_switch_threshold: float = 1.5,  # PCU hysteresis to prevent chatter
    ):
        self.phases = phases
        self.min_green_sec = min_green_sec
        self.max_green_sec = max_green_sec
        self.yellow_sec = yellow_sec
        self.all_red_sec = all_red_sec
        self.starvation_threshold_sec = starvation_threshold_sec
        self.pressure_switch_threshold = pressure_switch_threshold

        self.current_phase_index: int = 0
        self.current_signal_state: SignalState = SignalState.GREEN
        self.elapsed_in_state: int = 0
        self.target_phase_index: Optional[int] = None
        self.phase_pressures: Dict[str, float] = {}

    def reset(self):
        """Reset controller state to starting condition."""
        self.current_phase_index = 0
        self.current_signal_state = SignalState.GREEN
        self.elapsed_in_state = 0
        self.target_phase_index = None
        self.phase_pressures = {}

    @property
    def current_phase(self) -> PhaseConfig:
        return self.phases[self.current_phase_index]

    def _compute_phase_pressure(
        self, phase: PhaseConfig, approaches: Dict[str, ApproachState]
    ) -> float:
        """
        Calculates incoming queue pressure for a phase:
        Pressure = Sum of queues on served approaches (PCU).
        """
        pressure = 0.0
        for app_id in phase.served_approaches:
            if app_id in approaches:
                app = approaches[app_id]
                pressure += max(0.0, app.estimated_queue_pcu)
        return round(pressure, 2)

    def step(
        self,
        current_step: int,
        timestamp_str: str,
        approaches: Dict[str, ApproachState],
    ) -> ControllerDecision:
        """
        Advance one simulation second and produce the controller decision.
        """
        self.elapsed_in_state += 1

        # Calculate pressure for all available phases
        self.phase_pressures = {
            str(p.phase_id): self._compute_phase_pressure(p, approaches)
            for p in self.phases
        }

        # Check anti-starvation across all approaches
        starving_phase_idx: Optional[int] = None
        for idx, p in enumerate(self.phases):
            for app_id in p.served_approaches:
                app = approaches.get(app_id)
                if app and app.time_since_last_served_sec >= self.starvation_threshold_sec:
                    starving_phase_idx = idx
                    break
            if starving_phase_idx is not None:
                break

        # ── State Machine ──────────────────────────────────────────────────

        # 1. In GREEN
        if self.current_signal_state == SignalState.GREEN:
            curr_p = self.current_phase
            curr_pressure = self.phase_pressures.get(str(curr_p.phase_id), 0.0)

            # Determine best competing phase based on max pressure
            best_phase_idx = self.current_phase_index
            max_p = curr_pressure

            for idx, p in enumerate(self.phases):
                p_val = self.phase_pressures.get(str(p.phase_id), 0.0)
                if p_val > max_p + self.pressure_switch_threshold:
                    max_p = p_val
                    best_phase_idx = idx

            # Anti-starvation overrides max-pressure if another phase is starving
            if starving_phase_idx is not None and starving_phase_idx != self.current_phase_index:
                best_phase_idx = starving_phase_idx

            should_switch = False
            switch_rationale = ""

            # Check minimum green constraint
            if self.elapsed_in_state >= self.min_green_sec:
                # Max-green max-out
                if self.elapsed_in_state >= self.max_green_sec:
                    should_switch = True
                    # Round robin or best competing
                    next_idx = best_phase_idx if best_phase_idx != self.current_phase_index else (self.current_phase_index + 1) % len(self.phases)
                    self.target_phase_index = next_idx
                    switch_rationale = f"Max-out: reached {self.max_green_sec}s maximum green limit"
                # Competing phase has higher pressure or starving
                elif best_phase_idx != self.current_phase_index:
                    should_switch = True
                    self.target_phase_index = best_phase_idx
                    if starving_phase_idx == best_phase_idx:
                        switch_rationale = f"Anti-starvation override for Phase {self.phases[best_phase_idx].phase_id}"
                    else:
                        switch_rationale = (
                            f"Max-Pressure switch: Phase {self.phases[best_phase_idx].phase_id} "
                            f"(pressure {max_p:.1f} PCU) exceeds current Phase {curr_p.phase_id} ({curr_pressure:.1f} PCU)"
                        )

            if should_switch:
                self.current_signal_state = SignalState.YELLOW
                self.elapsed_in_state = 0
                return ControllerDecision(
                    step=current_step,
                    timestamp_str=timestamp_str,
                    mode=ControllerMode.MAX_PRESSURE,
                    active_phase_id=curr_p.phase_id,
                    phase_name=curr_p.name,
                    signal_state=SignalState.YELLOW,
                    time_remaining_in_state=self.yellow_sec,
                    elapsed_in_phase=self.elapsed_in_state,
                    phase_demand_scores=self.phase_pressures,
                    decision_rationale=f"Yellow clearance: {switch_rationale}",
                    is_transitioning=True,
                    transition_stage="YELLOW",
                )

            # Hold green
            time_rem = max(0, self.max_green_sec - self.elapsed_in_state)
            return ControllerDecision(
                step=current_step,
                timestamp_str=timestamp_str,
                mode=ControllerMode.MAX_PRESSURE,
                active_phase_id=curr_p.phase_id,
                phase_name=curr_p.name,
                signal_state=SignalState.GREEN,
                time_remaining_in_state=time_rem,
                elapsed_in_phase=self.elapsed_in_state,
                phase_demand_scores=self.phase_pressures,
                decision_rationale=f"Max-Pressure active: holding green on Phase {curr_p.phase_id} (pressure: {curr_pressure:.1f} PCU)",
                is_transitioning=False,
            )

        # 2. In YELLOW
        elif self.current_signal_state == SignalState.YELLOW:
            curr_p = self.current_phase
            if self.elapsed_in_state >= self.yellow_sec:
                self.current_signal_state = SignalState.ALL_RED
                self.elapsed_in_state = 0
                return ControllerDecision(
                    step=current_step,
                    timestamp_str=timestamp_str,
                    mode=ControllerMode.MAX_PRESSURE,
                    active_phase_id=curr_p.phase_id,
                    phase_name=curr_p.name,
                    signal_state=SignalState.ALL_RED,
                    time_remaining_in_state=self.all_red_sec,
                    elapsed_in_phase=self.elapsed_in_state,
                    phase_demand_scores=self.phase_pressures,
                    decision_rationale="All-red safety clearance interval",
                    is_transitioning=True,
                    transition_stage="ALL_RED",
                )
            time_rem = max(0, self.yellow_sec - self.elapsed_in_state)
            return ControllerDecision(
                step=current_step,
                timestamp_str=timestamp_str,
                mode=ControllerMode.MAX_PRESSURE,
                active_phase_id=curr_p.phase_id,
                phase_name=curr_p.name,
                signal_state=SignalState.YELLOW,
                time_remaining_in_state=time_rem,
                elapsed_in_phase=self.elapsed_in_state,
                phase_demand_scores=self.phase_pressures,
                decision_rationale="Yellow clearance in progress",
                is_transitioning=True,
                transition_stage="YELLOW",
            )

        # 3. In ALL_RED
        elif self.current_signal_state == SignalState.ALL_RED:
            if self.elapsed_in_state >= self.all_red_sec:
                # Transition to target phase
                if self.target_phase_index is not None:
                    self.current_phase_index = self.target_phase_index
                    self.target_phase_index = None
                else:
                    self.current_phase_index = (self.current_phase_index + 1) % len(self.phases)

                self.current_signal_state = SignalState.GREEN
                self.elapsed_in_state = 0
                new_p = self.current_phase
                return ControllerDecision(
                    step=current_step,
                    timestamp_str=timestamp_str,
                    mode=ControllerMode.MAX_PRESSURE,
                    active_phase_id=new_p.phase_id,
                    phase_name=new_p.name,
                    signal_state=SignalState.GREEN,
                    time_remaining_in_state=self.min_green_sec,
                    elapsed_in_phase=0,
                    phase_demand_scores=self.phase_pressures,
                    decision_rationale=f"Max-Pressure green initiated on {new_p.name}",
                    is_transitioning=False,
                )

            curr_p = self.current_phase
            time_rem = max(0, self.all_red_sec - self.elapsed_in_state)
            return ControllerDecision(
                step=current_step,
                timestamp_str=timestamp_str,
                mode=ControllerMode.MAX_PRESSURE,
                active_phase_id=curr_p.phase_id,
                phase_name=curr_p.name,
                signal_state=SignalState.ALL_RED,
                time_remaining_in_state=time_rem,
                elapsed_in_phase=self.elapsed_in_state,
                phase_demand_scores=self.phase_pressures,
                decision_rationale="All-red intersection clearance interval",
                is_transitioning=True,
                transition_stage="ALL_RED",
            )

        # Default fallback
        return ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=ControllerMode.MAX_PRESSURE,
            active_phase_id=self.current_phase.phase_id,
            phase_name=self.current_phase.name,
            signal_state=SignalState.GREEN,
            time_remaining_in_state=self.min_green_sec,
            elapsed_in_phase=self.elapsed_in_state,
            decision_rationale="Max-Pressure normal operation",
        )

