from typing import Dict, List, Optional
from backend.core.schema import (
    ApproachState,
    ControllerDecision,
    ControllerMode,
    PhaseConfig,
    SignalState,
)
from backend.core.config import (
    FIXED_CYCLE_SPLIT_SEC,
    YELLOW_CLEARANCE_SEC,
    ALL_RED_CLEARANCE_SEC,
)


class FixedTimeController:
    """
    Standard pre-timed fixed cycle signal controller.
    Cycles through configured phases with fixed green durations,
    strictly enforcing yellow and all-red clearance intervals.
    """

    def __init__(
        self,
        phases: List[PhaseConfig],
        fixed_green_sec: int = FIXED_CYCLE_SPLIT_SEC,
        yellow_sec: int = YELLOW_CLEARANCE_SEC,
        all_red_sec: int = ALL_RED_CLEARANCE_SEC,
    ):
        self.phases = phases
        self.fixed_green_sec = fixed_green_sec
        self.yellow_sec = yellow_sec
        self.all_red_sec = all_red_sec

        self.current_phase_index = 0
        self.current_signal_state = SignalState.GREEN
        self.elapsed_in_state = 0
        self.target_phase_index: Optional[int] = None

    def reset(self):
        self.current_phase_index = 0
        self.current_signal_state = SignalState.GREEN
        self.elapsed_in_state = 0
        self.target_phase_index = None

    @property
    def current_phase(self) -> PhaseConfig:
        return self.phases[self.current_phase_index]

    def step(
        self,
        current_step: int,
        timestamp_str: str,
        approaches: Dict[str, ApproachState],
    ) -> ControllerDecision:
        self.elapsed_in_state += 1
        active_phase = self.phases[self.current_phase_index]
        is_transitioning = self.current_signal_state in (SignalState.YELLOW, SignalState.ALL_RED)

        # State transition logic
        if self.current_signal_state == SignalState.GREEN:
            remaining = max(0, self.fixed_green_sec - self.elapsed_in_state)
            if self.elapsed_in_state >= self.fixed_green_sec:
                # Initiate transition: enter YELLOW
                self.current_signal_state = SignalState.YELLOW
                self.elapsed_in_state = 0
                self.target_phase_index = (self.current_phase_index + 1) % len(self.phases)
                rationale = f"Fixed green split ({self.fixed_green_sec}s) elapsed. Transitioning to Yellow clearance."
            else:
                rationale = f"Holding fixed green for {active_phase.name} ({remaining}s remaining)."

        elif self.current_signal_state == SignalState.YELLOW:
            remaining = max(0, self.yellow_sec - self.elapsed_in_state)
            if self.elapsed_in_state >= self.yellow_sec:
                # Enter ALL_RED clearance
                self.current_signal_state = SignalState.ALL_RED
                self.elapsed_in_state = 0
                rationale = f"Yellow clearance ({self.yellow_sec}s) complete. Entering All-Red clearance."
            else:
                rationale = f"Yellow clearance in progress ({remaining}s remaining)."

        elif self.current_signal_state == SignalState.ALL_RED:
            remaining = max(0, self.all_red_sec - self.elapsed_in_state)
            if self.elapsed_in_state >= self.all_red_sec:
                # Switch to next phase GREEN
                self.current_phase_index = self.target_phase_index
                self.current_signal_state = SignalState.GREEN
                self.elapsed_in_state = 0
                self.target_phase_index = None
                active_phase = self.phases[self.current_phase_index]
                rationale = f"All-Red clearance complete. Activating green for {active_phase.name}."
            else:
                rationale = f"All-Red junction clearance interval active ({remaining}s remaining)."

        return ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=ControllerMode.FIXED_TIME,
            active_phase_id=active_phase.phase_id,
            phase_name=active_phase.name,
            signal_state=self.current_signal_state,
            time_remaining_in_state=max(0, (
                self.fixed_green_sec if self.current_signal_state == SignalState.GREEN
                else self.yellow_sec if self.current_signal_state == SignalState.YELLOW
                else self.all_red_sec
            ) - self.elapsed_in_state),
            elapsed_in_phase=self.elapsed_in_state,
            phase_demand_scores={},
            decision_rationale=rationale,
            is_transitioning=is_transitioning,
            transition_stage=self.current_signal_state.value if is_transitioning else None,
        )

