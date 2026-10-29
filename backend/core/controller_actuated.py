"""
Actuated Signal Controller.

A standard demand-responsive actuated controller that extends or terminates
the green indication based on observed vehicle detections / gaps, bounded by
configurable minimum and maximum green times.
"""

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
)


class ActuatedController:
    """
    Traffic-Actuated Signal Controller (Vehicle-Actuated / Gap-Out Logic).

    How it works:
    - Starts in GREEN with at least `min_green` seconds.
    - While in GREEN, if detected vehicle arrivals or queue on the active phase
      exceed the passage/extension threshold (`gap_threshold`), green is extended
      by `unit_extension_sec` up to `max_green`.
    - If demand drops below the threshold (gap-out) or reaches `max_green` (max-out),
      the controller transitions through YELLOW and ALL-RED clearance intervals.
    - Next phase is selected based on which waiting phase has pending demand
      (with anti-starvation round-robin fallback).
    """

    def __init__(
        self,
        phases: List[PhaseConfig],
        min_green_sec: int = MIN_GREEN_SEC,
        max_green_sec: int = MAX_GREEN_SEC,
        yellow_sec: int = YELLOW_CLEARANCE_SEC,
        all_red_sec: int = ALL_RED_CLEARANCE_SEC,
        gap_threshold_pcu: float = 0.3,  # arrivals/sec needed to maintain green
        unit_extension_sec: int = 3,
    ):
        self.phases = phases
        self.min_green_sec = min_green_sec
        self.max_green_sec = max_green_sec
        self.yellow_sec = yellow_sec
        self.all_red_sec = all_red_sec
        self.gap_threshold_pcu = gap_threshold_pcu
        self.unit_extension_sec = unit_extension_sec

        self.current_phase_index = 0
        self.current_signal_state = SignalState.GREEN
        self.elapsed_in_state = 0
        self.target_phase_index: Optional[int] = None
        self.green_extension_remaining = 0
        self.termination_reason = ""

    def reset(self):
        self.current_phase_index = 0
        self.current_signal_state = SignalState.GREEN
        self.elapsed_in_state = 0
        self.target_phase_index = None
        self.green_extension_remaining = 0
        self.termination_reason = ""

    @property
    def current_phase(self) -> PhaseConfig:
        return self.phases[self.current_phase_index]

    def _select_next_phase(self, approaches: Dict[str, ApproachState]) -> int:
        """Select next phase with pending demand; round-robin if multiple or none."""
        best_idx = (self.current_phase_index + 1) % len(self.phases)
        highest_demand = -1.0

        for i, phase in enumerate(self.phases):
            if i == self.current_phase_index:
                continue
            demand = sum(
                approaches[app_id].estimated_queue_pcu
                + approaches[app_id].arrival_rate_pcu_per_min / 60.0
                for app_id in phase.served_approaches
                if app_id in approaches
            )
            if demand > highest_demand:
                highest_demand = demand
                best_idx = i

        return best_idx

    def step(
        self,
        current_step: int,
        timestamp_str: str,
        approaches: Dict[str, ApproachState],
    ) -> ControllerDecision:
        self.elapsed_in_state += 1
        active_phase = self.phases[self.current_phase_index]
        is_transitioning = self.current_signal_state in (SignalState.YELLOW, SignalState.ALL_RED)
        rationale = ""

        if self.current_signal_state == SignalState.GREEN:
            # Active phase demand check
            active_arrivals = sum(
                approaches[app_id].arrivals_last_step.to_pcu()
                for app_id in active_phase.served_approaches
                if app_id in approaches
            )
            active_queue = sum(
                approaches[app_id].estimated_queue_pcu
                for app_id in active_phase.served_approaches
                if app_id in approaches
            )

            # Check if vehicles are arriving to extend green
            if active_arrivals >= self.gap_threshold_pcu or active_queue > 1.0:
                self.green_extension_remaining = min(
                    self.unit_extension_sec,
                    self.max_green_sec - self.elapsed_in_state,
                )

            # Decision to terminate green or extend
            has_reached_min = self.elapsed_in_state >= self.min_green_sec
            has_reached_max = self.elapsed_in_state >= self.max_green_sec
            has_gapped_out = has_reached_min and self.green_extension_remaining <= 0 and active_queue < 0.5

            if has_reached_max:
                self.termination_reason = "Max-out: reached maximum green limit."
                self.current_signal_state = SignalState.YELLOW
                self.elapsed_in_state = 0
                self.target_phase_index = self._select_next_phase(approaches)
                rationale = f"Actuated controller {self.termination_reason} Transitioning to Yellow."
            elif has_gapped_out:
                self.termination_reason = "Gap-out: vehicle headway exceeded threshold."
                self.current_signal_state = SignalState.YELLOW
                self.elapsed_in_state = 0
                self.target_phase_index = self._select_next_phase(approaches)
                rationale = f"Actuated controller {self.termination_reason} Transitioning to Yellow."
            else:
                if self.green_extension_remaining > 0:
                    self.green_extension_remaining -= 1
                remaining_min = max(0, self.min_green_sec - self.elapsed_in_state)
                remaining_max = max(0, self.max_green_sec - self.elapsed_in_state)
                rationale = (
                    f"Holding actuated green for {active_phase.name}. "
                    f"Min green left: {remaining_min}s, Max green left: {remaining_max}s."
                )

        elif self.current_signal_state == SignalState.YELLOW:
            remaining = max(0, self.yellow_sec - self.elapsed_in_state)
            if self.elapsed_in_state >= self.yellow_sec:
                self.current_signal_state = SignalState.ALL_RED
                self.elapsed_in_state = 0
                rationale = f"Yellow clearance ({self.yellow_sec}s) complete. Entering All-Red clearance."
            else:
                rationale = f"Yellow clearance in progress ({remaining}s remaining)."

        elif self.current_signal_state == SignalState.ALL_RED:
            remaining = max(0, self.all_red_sec - self.elapsed_in_state)
            if self.elapsed_in_state >= self.all_red_sec:
                self.current_phase_index = self.target_phase_index or ((self.current_phase_index + 1) % len(self.phases))
                self.current_signal_state = SignalState.GREEN
                self.elapsed_in_state = 0
                self.target_phase_index = None
                self.green_extension_remaining = 0
                active_phase = self.phases[self.current_phase_index]
                rationale = f"All-Red complete. Activating green for {active_phase.name}."
            else:
                rationale = f"All-Red junction clearance active ({remaining}s remaining)."

        return ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=ControllerMode.ACTUATED,
            active_phase_id=active_phase.phase_id,
            phase_name=active_phase.name,
            signal_state=self.current_signal_state,
            time_remaining_in_state=max(0, (
                self.max_green_sec if self.current_signal_state == SignalState.GREEN
                else self.yellow_sec if self.current_signal_state == SignalState.YELLOW
                else self.all_red_sec
            ) - self.elapsed_in_state),
            elapsed_in_phase=self.elapsed_in_state,
            phase_demand_scores={},
            decision_rationale=rationale,
            is_transitioning=is_transitioning,
            transition_stage=self.current_signal_state.value if is_transitioning else None,
        )

