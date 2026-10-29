"""
Shared application state — a singleton that holds the running simulation,
controller, and last evaluation report so all API endpoints share it.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from backend.core.schema import (
    ControllerDecision, ControllerMode, EvaluationReport,
    SafetyEvent, StepObservation,
)
from backend.core.config import FIXED_CYCLE_SPLIT_SEC
from backend.core.controller_baseline import FixedTimeController
from backend.core.controller_adaptive import AdaptiveController
from backend.core.controller_actuated import ActuatedController
from backend.core.safety_layer import SafetySupervisor
from backend.data.loader import DatasetAdapter
from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.replay_engine import JunctionSimulator

logger = logging.getLogger(__name__)


class AppState:
    """Central state container for the running API service."""

    def __init__(self):
        self.adapter = DatasetAdapter()
        self.junction_id: str = "AlankarChowk"
        self.date: str = "11"
        self.mode: ControllerMode = ControllerMode.ADAPTIVE
        self.layout: Optional[JunctionLayout] = None
        self.fixed_ctrl: Optional[FixedTimeController] = None
        self.actuated_ctrl: Optional[ActuatedController] = None
        self.adaptive_ctrl: Optional[AdaptiveController] = None
        self.safety_supervisor: Optional[SafetySupervisor] = None
        self.sim: Optional[JunctionSimulator] = None
        self.last_decision: Optional[ControllerDecision] = None
        self.last_observation: Optional[StepObservation] = None
        self.history: List[StepObservation] = []
        self.decision_log: List[ControllerDecision] = []
        self.safety_events: List[SafetyEvent] = []
        self.is_running: bool = False
        self.is_paused: bool = False
        self.evaluation_report: Optional[EvaluationReport] = None
        self._lock = asyncio.Lock()
        self._initialized: bool = False

    def initialize(
        self,
        junction_id: str = "AlankarChowk",
        date: str = "11",
        duration_sec: int = 1800,
        mode: ControllerMode = ControllerMode.ADAPTIVE,
    ):
        """Load dataset and set up controllers and simulator."""
        self.junction_id = junction_id
        self.date = date
        self.mode = mode
        self.layout = JunctionLayout(junction_id)

        # Determine needed cameras from layout
        cameras = set(meta["camera_id"] for meta in self.layout.approaches.values())
        primary_cam = sorted(cameras)[0]

        arrivals_df = self.adapter.load_camera_arrivals(
            junction_id, primary_cam, date, max_seconds=duration_sec + 120
        )

        self.fixed_ctrl = FixedTimeController(
            phases=self.layout.phases, fixed_green_sec=FIXED_CYCLE_SPLIT_SEC
        )
        self.actuated_ctrl = ActuatedController(phases=self.layout.phases)
        self.adaptive_ctrl = AdaptiveController(phases=self.layout.phases)
        self.safety_supervisor = SafetySupervisor(
            intersection_id=junction_id,
            phases=self.layout.phases,
        )
        self.sim = JunctionSimulator(
            layout=self.layout, arrival_data=arrivals_df, date=date
        )

        self.last_decision = None
        self.last_observation = None
        self.history = []
        self.decision_log = []
        self.safety_events = []
        self.is_running = False
        self.is_paused = False
        self._initialized = True
        logger.info(f"AppState initialized: {junction_id}, Jan-{date}, mode={mode.value}")

    def _get_active_controller(self):
        """Return the currently active controller based on mode."""
        if self.mode == ControllerMode.FIXED_TIME:
            return self.fixed_ctrl
        elif self.mode == ControllerMode.ACTUATED:
            return self.actuated_ctrl
        else:
            return self.adaptive_ctrl

    def step_once(self) -> Optional[StepObservation]:
        if not self._initialized or self.sim is None:
            return None
        if self.sim.step_count >= self.sim.total_steps:
            self.is_running = False
            return None

        ctrl = self._get_active_controller()
        ts = self.sim.time_index[self.sim.step_count] if self.sim.step_count < len(self.sim.time_index) else ""

        decision = ctrl.step(
            current_step=self.sim.step_count,
            timestamp_str=ts,
            approaches=self.sim.approaches,
        )

        # Safety supervisor validates and may override the decision
        if self.safety_supervisor is not None:
            decision, events = self.safety_supervisor.validate_and_enforce(
                decision=decision,
                approaches=self.sim.approaches,
                current_step=self.sim.step_count,
                timestamp_str=ts,
            )
            self.safety_events.extend(events)

        obs = self.sim.step(decision)
        self.last_decision = decision
        self.last_observation = obs
        self.history.append(obs)
        self.decision_log.append(decision)
        return obs

    def reset(self):
        if not self._initialized:
            return
        if self.fixed_ctrl:
            self.fixed_ctrl.reset()
        if self.actuated_ctrl:
            self.actuated_ctrl.reset()
        if self.adaptive_ctrl:
            self.adaptive_ctrl.reset()
        if self.safety_supervisor:
            self.safety_supervisor.reset()
        if self.sim:
            self.sim.reset()
        self.history = []
        self.decision_log = []
        self.safety_events = []
        self.last_decision = None
        self.last_observation = None
        self.is_running = False
        self.is_paused = False


# Global singleton
app_state = AppState()
