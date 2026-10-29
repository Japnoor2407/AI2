"""
Dataset Replay Simulation Engine.

Reads per-second arrival increments from the real Pune CCTV dataset and drives
a discrete-time junction simulation. Both the fixed-time and adaptive
controllers are fed identical demand streams so results are directly comparable.

Queue Model (per approach, per second):
    Q(t+1) = max(0, Q(t) + arrivals(t) - departures(t))
where:
    departures(t) = SAT_FLOW * green_flag(t)  [PCU/sec discharged during green]
    
Waiting time estimation (Little's Law proxy):
    W(t) = Q(t) / max(arrival_rate, epsilon)
"""

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import pandas as pd

from backend.core.schema import (
    ApproachState, ControllerDecision, ControllerMode,
    SignalState, StepObservation, VehicleClassCounts,
)
from backend.core.config import PCU_WEIGHTS, STARVATION_THRESHOLD_SEC
from backend.simulation.junction_layout import JunctionLayout

logger = logging.getLogger(__name__)

RATE_WINDOW_SEC = 60  # rolling window for arrival rate estimation


class JunctionSimulator:
    """
    Discrete-time (1-second step) junction simulation driven by real arrival data.
    """

    def __init__(
        self,
        layout: JunctionLayout,
        arrival_data: pd.DataFrame,
        date: str = "11",
    ):
        self.layout = layout
        self.arrival_data = arrival_data
        self.date = date

        # Index arrivals by (time_str, direction)
        self.time_index: List[str] = []
        self.arrival_lookup: Dict[Tuple[str, str], VehicleClassCounts] = {}
        self._build_index()

        # Approach runtime state
        self.approaches: Dict[str, ApproachState] = {}
        self._arrival_history: Dict[str, Deque[float]] = {}
        self.step_count = 0
        self.log: List[StepObservation] = []
        self._total_throughput_pcu: float = 0.0

        self._init_approaches()

    def _build_index(self):
        if self.arrival_data is None or self.arrival_data.empty:
            return
        times = self.arrival_data["Time"].drop_duplicates().tolist()
        self.time_index = times

        # Map (time, direction) -> arrivals for all four vehicle classes
        arr_cols = [c for c in ["car_arrivals", "motorbike_arrivals", "bus_arrivals", "truck_arrivals"]
                    if c in self.arrival_data.columns]
        has_arr = len(arr_cols) == 4

        for _, row in self.arrival_data.iterrows():
            key = (str(row["Time"]), str(row["Direction"]))
            if has_arr:
                self.arrival_lookup[key] = VehicleClassCounts(
                    car=int(row.get("car_arrivals", 0)),
                    motorbike=int(row.get("motorbike_arrivals", 0)),
                    bus=int(row.get("bus_arrivals", 0)),
                    truck=int(row.get("truck_arrivals", 0)),
                )

    def _init_approaches(self):
        for app_id, meta in self.layout.approaches.items():
            self.approaches[app_id] = ApproachState(
                approach_id=app_id,
                approach_name=meta["name"],
                direction=meta["direction"],
                camera_id=meta["camera_id"],
                junction_id=self.layout.junction_id,
                saturation_flow_rate_pcu_per_sec=meta.get("saturation_flow", 0.6),
            )
            self._arrival_history[app_id] = deque(maxlen=RATE_WINDOW_SEC)

    def reset(self):
        self.step_count = 0
        self.log = []
        self._total_throughput_pcu = 0.0
        self._init_approaches()

    @property
    def total_steps(self) -> int:
        return len(self.time_index)

    def _get_arrivals(self, time_str: str, direction: str) -> VehicleClassCounts:
        return self.arrival_lookup.get((time_str, direction), VehicleClassCounts())

    def _get_current_phase_served(self, decision: ControllerDecision) -> List[str]:
        phase_map = {p.phase_id: p for p in self.layout.phases}
        phase = phase_map.get(decision.active_phase_id)
        if phase is None:
            return []
        if decision.signal_state == SignalState.GREEN:
            return phase.served_approaches
        return []

    def step(self, decision: ControllerDecision) -> StepObservation:
        if self.step_count >= len(self.time_index):
            raise StopIteration("Dataset replay exhausted")

        current_time_str = self.time_index[self.step_count]
        served_approaches = self._get_current_phase_served(decision)

        # Process each approach
        for app_id, app in self.approaches.items():
            meta = self.layout.approaches[app_id]
            direction = meta["direction"]

            # Look up real arrivals from dataset
            arrivals = self._get_arrivals(current_time_str, direction)
            arrival_pcu = arrivals.to_pcu()
            self._arrival_history[app_id].append(arrival_pcu)

            # Rolling arrival rate (PCU/min)
            hist = self._arrival_history[app_id]
            elapsed = max(1, len(hist))
            arrival_rate_per_min = (sum(hist) / elapsed) * 60.0

            # Queue update
            is_green = app_id in served_approaches
            discharge = app.saturation_flow_rate_pcu_per_sec if is_green else 0.0
            old_queue = app.estimated_queue_pcu
            new_queue = max(0.0, old_queue + arrival_pcu - discharge)

            # Throughput
            actual_discharge = min(old_queue + arrival_pcu, discharge)
            self._total_throughput_pcu += actual_discharge

            # Waiting time estimate via Little's Law: W = Q / lambda
            lam = max(arrival_rate_per_min / 60.0, 0.001)
            avg_wait = new_queue / lam

            # Time since last served
            if is_green:
                app.time_since_last_served_sec = 0.0
            else:
                app.time_since_last_served_sec += 1.0

            is_starving = app.time_since_last_served_sec >= STARVATION_THRESHOLD_SEC

            # Update cumulative stats
            app.arrivals_last_step = arrivals
            app.cumulative_arrivals_pcu += arrival_pcu
            app.cumulative_departures_pcu += actual_discharge
            app.estimated_queue_pcu = round(new_queue, 2)
            app.estimated_queue_vehicles = int(new_queue / 1.2)  # approx PCU→vehicles
            app.arrival_rate_pcu_per_min = round(arrival_rate_per_min, 3)
            app.average_wait_time_sec = round(avg_wait, 2)
            app.max_wait_time_sec = max(app.max_wait_time_sec, avg_wait)
            # Approach signal state logic
            if is_green:
                if decision.signal_state == SignalState.YELLOW:
                    app.current_signal = SignalState.YELLOW
                elif decision.signal_state == SignalState.ALL_RED:
                    app.current_signal = SignalState.ALL_RED
                else:
                    app.current_signal = SignalState.GREEN
            else:
                if decision.signal_state == SignalState.ALL_RED:
                    app.current_signal = SignalState.ALL_RED
                else:
                    app.current_signal = SignalState.RED
            app.is_starving = is_starving

        total_queue = sum(a.estimated_queue_pcu for a in self.approaches.values())
        avg_delay = sum(a.average_wait_time_sec for a in self.approaches.values()) / max(1, len(self.approaches))

        obs = StepObservation(
            step=self.step_count,
            timestamp_str=current_time_str,
            controller_decision=decision,
            approaches={k: v.model_copy() for k, v in self.approaches.items()},
            total_active_queue_pcu=round(total_queue, 2),
            total_throughput_pcu=round(self._total_throughput_pcu, 2),
            average_delay_sec=round(avg_delay, 2),
        )
        self.log.append(obs)
        self.step_count += 1
        return obs

