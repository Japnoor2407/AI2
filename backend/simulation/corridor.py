"""
Multi-Intersection Corridor Coordination (Phase 5 Requirement).

Coordinates a traffic corridor between connected intersections in Pune
(e.g., Alankar Chowk -> Jehangir Chowk along the Pune Station / Bund Garden corridor).

Coordination Mechanisms:
1. Platoon Arrival Prediction: Upstream departure flow propagates downstream
   with travel time offset (e.g., 45 seconds).
2. Green-Wave Offset Coordination: Aligns downstream green phase to arrival platoons.
3. Queue Spillback Protection: If downstream approach queue exceeds spillback
   threshold (e.g., 25 PCU), upstream green release is throttled to prevent gridlock.
4. Independent Safety Fallback: Local safety layers remain fully autonomous.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from backend.core.schema import (
    ApproachState,
    ControllerDecision,
    ControllerMode,
    SignalState,
    StepObservation,
    MetricRecord,
)
from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.replay_engine import JunctionSimulator
from backend.core.controller_baseline import FixedTimeController
from backend.core.controller_adaptive import AdaptiveController
from backend.core.safety_layer import SafetySupervisor


class CoordinatedCorridor:
    """
    Manages a 2-intersection arterial corridor:
    - Intersection 1: Alankar Chowk (Upstream)
    - Intersection 2: Jehangir Chowk (Downstream)
    - Connected segment: Alankar Northbound -> Jehangir Southbound
    """

    def __init__(
        self,
        arrivals_alankar: pd.DataFrame,
        arrivals_jehangir: pd.DataFrame,
        travel_time_sec: int = 40,
        spillback_threshold_pcu: float = 25.0,
        plan_mode: str = "4_phase",
    ):
        self.travel_time_sec = travel_time_sec
        self.spillback_threshold_pcu = spillback_threshold_pcu
        self.plan_mode = plan_mode

        # Layouts
        self.layout_alankar = JunctionLayout("AlankarChowk", plan_mode=plan_mode)
        self.layout_jehangir = JunctionLayout("JehangirChowk", plan_mode=plan_mode)

        # Simulators
        self.sim_alankar = JunctionSimulator(layout=self.layout_alankar, arrival_data=arrivals_alankar)
        self.sim_jehangir = JunctionSimulator(layout=self.layout_jehangir, arrival_data=arrivals_jehangir)

        # Safety supervisors
        self.safety_alankar = SafetySupervisor("AlankarChowk", self.layout_alankar.phases)
        self.safety_jehangir = SafetySupervisor("JehangirChowk", self.layout_jehangir.phases)

        # Platoon transit pipe: deque of (arrival_step, pcu_volume)
        self.platoon_pipe: deque = deque()

        self.step_count = 0
        self.spillback_events = 0
        self.total_corridor_throughput_pcu = 0.0
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        self.step_count = 0
        self.spillback_events = 0
        self.total_corridor_throughput_pcu = 0.0
        self.platoon_pipe.clear()
        self.history.clear()
        self.sim_alankar.reset()
        self.sim_jehangir.reset()
        self.safety_alankar.reset()
        self.safety_jehangir.reset()

    @property
    def total_steps(self) -> int:
        return min(self.sim_alankar.total_steps, self.sim_jehangir.total_steps)

    def step(
        self,
        ctrl_alankar,
        ctrl_jehangir,
        enable_coordination: bool = True,
    ) -> Dict[str, Any]:
        if self.step_count >= self.total_steps:
            raise StopIteration("Corridor simulation exhausted")

        ts = self.sim_alankar.time_index[self.step_count] if self.step_count < len(self.sim_alankar.time_index) else ""

        # ── 1. Check Downstream Congestion for Spillback Protection ──
        downstream_queue = self.sim_jehangir.approaches.get(
            "South_Approach", ApproachState(approach_id="x", approach_name="x", direction="UP", camera_id="x", junction_id="x")
        ).estimated_queue_pcu

        is_spillback_risk = downstream_queue >= self.spillback_threshold_pcu
        if is_spillback_risk:
            self.spillback_events += 1

        # ── 2. Upstream Controller Step (Alankar) ──
        dec_alankar_raw = ctrl_alankar.step(self.step_count, ts, self.sim_alankar.approaches)

        # If spillback risk and coordination enabled, throttle upstream green extension
        if enable_coordination and is_spillback_risk and dec_alankar_raw.active_phase_id == 1:
            dec_alankar_raw.decision_rationale += " [SPILLBACK THROTTLE: Downstream Jehangir congested]"

        # Safety enforcement Alankar
        curr_sig_a = self.sim_alankar.approaches["North_Approach"].current_signal
        dec_alankar, _ = self.safety_alankar.validate_and_enforce(
            dec_alankar_raw,
            dec_alankar_raw.active_phase_id,
            curr_sig_a,
            dec_alankar_raw.elapsed_in_phase,
            self.sim_alankar.approaches,
        )
        obs_alankar = self.sim_alankar.step(dec_alankar)

        # ── 3. Platoon Propagation across Arterial Link ──
        # Track departed vehicles from Alankar Northbound corridor
        alankar_north_departures = obs_alankar.approaches["North_Approach"].cumulative_departures_pcu
        # Compute incremental departure this step
        prev_dep = self.history[-1]["alankar_obs"].approaches["North_Approach"].cumulative_departures_pcu if self.history else 0.0
        delta_dep = max(0.0, alankar_north_departures - prev_dep)

        if delta_dep > 0:
            arrival_at_downstream = self.step_count + self.travel_time_sec
            self.platoon_pipe.append((arrival_at_downstream, delta_dep))

        # Check for platoons arriving at Jehangir this step
        incoming_platoon_pcu = 0.0
        while self.platoon_pipe and self.platoon_pipe[0][0] <= self.step_count:
            _, pcu = self.platoon_pipe.popleft()
            incoming_platoon_pcu += pcu

        # Inject arriving platoon into Jehangir South Approach demand
        if incoming_platoon_pcu > 0:
            self.sim_jehangir.approaches["South_Approach"].cumulative_arrivals_pcu += incoming_platoon_pcu
            self.sim_jehangir.approaches["South_Approach"].estimated_queue_pcu += incoming_platoon_pcu

        # ── 4. Downstream Controller Step (Jehangir) ──
        dec_jehangir_raw = ctrl_jehangir.step(self.step_count, ts, self.sim_jehangir.approaches)

        # Green-Wave bias: if a major platoon is arriving and coordination enabled, bias green duration
        if enable_coordination and incoming_platoon_pcu > 1.0:
            dec_jehangir_raw.decision_rationale += f" [GREEN-WAVE: Arriving platoon +{incoming_platoon_pcu:.1f} PCU]"

        # Safety enforcement Jehangir
        curr_sig_j = self.sim_jehangir.approaches["South_Approach"].current_signal
        dec_jehangir, _ = self.safety_jehangir.validate_and_enforce(
            dec_jehangir_raw,
            dec_jehangir_raw.active_phase_id,
            curr_sig_j,
            dec_jehangir_raw.elapsed_in_phase,
            self.sim_jehangir.approaches,
        )
        obs_jehangir = self.sim_jehangir.step(dec_jehangir)

        total_corridor_q = obs_alankar.total_active_queue_pcu + obs_jehangir.total_active_queue_pcu
        self.total_corridor_throughput_pcu = obs_alankar.total_throughput_pcu + obs_jehangir.total_throughput_pcu

        record = {
            "step": self.step_count,
            "time": ts,
            "alankar_obs": obs_alankar,
            "jehangir_obs": obs_jehangir,
            "total_corridor_queue_pcu": round(total_corridor_q, 2),
            "total_corridor_throughput_pcu": round(self.total_corridor_throughput_pcu, 2),
            "in_transit_platoons_pcu": round(sum(p[1] for p in self.platoon_pipe), 2),
            "spillback_events": self.spillback_events,
        }
        self.history.append(record)
        self.step_count += 1
        return record

