"""
SUMO and TraCI Simulation Adapter for TrafficSignalEnv.

Provides a unified simulation interface supporting:
1. SUMO via TraCI (microscopic individual-vehicle simulation).
2. JunctionSimulator (discrete-time empirical replay engine as a robust fallback).

SUMO is an optional dependency: if the SUMO binary is absent or TraCI encounters
a connection issue, the adapter transparently falls back to JunctionSimulator
without interrupting the RL environment or API server.
"""

import logging
import os
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional, Tuple

from backend.core.schema import (
    ApproachState,
    ControllerDecision,
    SignalState,
    StepObservation,
    VehicleClassCounts,
)
from backend.core.config import STARVATION_THRESHOLD_SEC
from backend.simulation.replay_engine import JunctionSimulator
from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.sumo.network_generator import network_generator
from backend.simulation.sumo.demand_generator import demand_generator

logger = logging.getLogger(__name__)

# Approach to SUMO incoming edge lane mapping
APPROACH_SUMO_LANES: Dict[str, List[str]] = {
    "North_Approach": ["N2C_0", "N2C_1"],
    "South_Approach": ["S2C_0", "S2C_1"],
    "East_Approach": ["E2C_0", "E2C_1"],
    "West_Approach": ["W2C_0", "W2C_1"],
}

# Signal state strings for TLS J_center (16 links: 4 per incoming approach)
TLS_ID = "J_center"
TLS_STATES = {
    # Phase 1: North-South Green
    (1, SignalState.GREEN):   "GGGGrrrrGGGGrrrr",
    # Phase 1: North-South Yellow Clearance
    (1, SignalState.YELLOW):  "yyyyrrrryyyyrrrr",
    # Phase 2: East-West Green
    (2, SignalState.GREEN):   "rrrrGGGGrrrrGGGG",
    # Phase 2: East-West Yellow Clearance
    (2, SignalState.YELLOW):  "rrrryyyyrrrryyyy",
    # All-Red Clearance (all approaches red)
    "ALL_RED":                "rrrrrrrrrrrrrrrr",
}


def is_sumo_available() -> bool:
    """Return True if a SUMO executable is discoverable on PATH."""
    return bool(
        shutil.which("sumo")
        or shutil.which("sumo-gui")
        or shutil.which("sumo.exe")
        or shutil.which("sumo-gui.exe")
    )


class SimulationAdapter:
    """
    Unified simulation interface. Exposes identical properties and methods
    to TrafficSignalEnv regardless of whether SUMO or JunctionSimulator is active.
    """

    def __init__(
        self,
        layout: JunctionLayout,
        arrival_data=None,
        date: str = "11",
        use_sumo: Optional[bool] = None,
        sumo_gui: bool = False,
    ):
        self.layout = layout
        self.date = date
        self.sumo_gui = sumo_gui
        self.arrival_data = self._prepare_arrivals(arrival_data)

        self._backend: str = "junction_sim"
        self._sim: Optional[JunctionSimulator] = None
        self._traci = None
        self._sumo_proc = None
        self._step_count: int = 0
        self._total_steps: int = 1800
        self._time_index: List[str] = []
        self._approaches: Dict[str, ApproachState] = {}
        self._sumo_throughput_pcu: float = 0.0
        self.log: List[StepObservation] = []

        # Determine backend
        should_use_sumo = use_sumo if use_sumo is not None else is_sumo_available()
        if should_use_sumo:
            try:
                self._init_sumo()
                self._backend = "sumo"
                logger.info("SimulationAdapter: using microscopic SUMO+TraCI backend.")
            except Exception as exc:
                logger.warning(
                    f"SimulationAdapter: SUMO initialization failed ({exc}). "
                    "Falling back to JunctionSimulator."
                )
                self._init_junction_sim()
                self._backend = "junction_sim"
        else:
            self._init_junction_sim()
            self._backend = "junction_sim"
            logger.info("SimulationAdapter: using JunctionSimulator fallback (SUMO unavailable or disabled).")

    def _prepare_arrivals(self, arrival_data):
        """Load empirical dataset or generate calibrated baseline synthetic arrivals."""
        if arrival_data is not None and not (hasattr(arrival_data, "empty") and arrival_data.empty):
            return arrival_data

        try:
            from backend.data.loader import DatasetAdapter
            adapter = DatasetAdapter()
            cameras = set(m.get("camera_id", "a2") for m in self.layout.approaches.values())
            primary_cam = sorted(cameras)[0] if cameras else "a2"
            df = adapter.load_camera_arrivals(
                self.layout.junction_id, primary_cam, str(self.date), max_seconds=1800
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.debug(f"Could not load empirical arrivals in SimulationAdapter: {e}")

        return self._create_synthetic_arrivals()

    def _create_synthetic_arrivals(self):
        """Creates realistic baseline arrivals when dataset CSV is missing."""
        import pandas as pd
        import numpy as np
        rows = []
        directions = ["UP", "DOWN", "RIGHT", "LEFT"]
        for s in range(1800):
            total_sec = 9 * 3600 + s
            h = (total_sec // 3600) % 24
            m = (total_sec % 3600) // 60
            sec = total_sec % 60
            time_str = f"{h:02d}:{m:02d}:{sec:02d}"
            for d in directions:
                base = 0.35 if d in ["UP", "DOWN"] else 0.2
                cars = 1 if np.random.rand() < base else 0
                bikes = 1 if np.random.rand() < (base * 1.5) else 0
                rows.append({
                    "Time": time_str,
                    "Direction": d,
                    "car_arrivals": cars,
                    "motorbike_arrivals": bikes,
                    "bus_arrivals": 0,
                    "truck_arrivals": 0,
                })
        return pd.DataFrame(rows)

    # ──────────────────────────────────────────────────────────
    # JunctionSimulator Fallback Backend
    # ──────────────────────────────────────────────────────────

    def _init_junction_sim(self):
        self._sim = JunctionSimulator(
            layout=self.layout,
            arrival_data=self.arrival_data,
            date=self.date,
        )
        self._step_count = self._sim.step_count
        self._total_steps = self._sim.total_steps
        self._time_index = self._sim.time_index
        self._approaches = self._sim.approaches
        self.log = self._sim.log

    # ──────────────────────────────────────────────────────────
    # SUMO + TraCI Backend
    # ──────────────────────────────────────────────────────────

    def _init_sumo(self):
        import traci

        sumo_bin = (
            shutil.which("sumo-gui") if self.sumo_gui else None
        ) or shutil.which("sumo") or shutil.which("sumo.exe")

        if not sumo_bin:
            raise FileNotFoundError("SUMO binary not found on system PATH.")

        # Ensure SUMO files exist
        sumo_dir = Path(__file__).parent.parent / "simulation" / "sumo"
        net_path = sumo_dir / "alankar_chowk.net.xml"
        sumocfg_path = sumo_dir / "alankar_chowk.sumocfg"

        if not net_path.exists():
            network_generator.generate_all()

        # Generate routes from arrival data
        demand_generator.generate_routes_xml(self.arrival_data, max_duration_sec=self._total_steps)

        # Start TraCI
        cmd = [
            sumo_bin,
            "-c", str(sumocfg_path),
            "--no-warnings", "true",
            "--step-length", "1.0",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
        ]

        traci.start(cmd)
        self._traci = traci

        # Initialise state
        self._step_count = 0
        self._total_steps = 1800
        self._sumo_throughput_pcu = 0.0
        self._time_index = [
            f"{9 + (s // 3600):02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
            for s in range(self._total_steps)
        ]
        self.log = []

        # Build initial approach states
        self._approaches = {}
        for app_id, meta in self.layout.approaches.items():
            self._approaches[app_id] = ApproachState(
                approach_id=app_id,
                approach_name=meta["name"],
                direction=meta["direction"],
                camera_id=meta.get("camera_id", "a2"),
                junction_id=self.layout.junction_id,
                current_signal=SignalState.RED,
                saturation_flow_rate_pcu_per_sec=meta.get("saturation_flow", 0.6),
            )

    # ──────────────────────────────────────────────────────────
    # Unified Public Properties
    # ──────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Returns 'sumo' or 'junction_sim'."""
        return self._backend

    @property
    def step_count(self) -> int:
        if self._backend == "junction_sim" and self._sim is not None:
            return self._sim.step_count
        return self._step_count

    @property
    def total_steps(self) -> int:
        if self._backend == "junction_sim" and self._sim is not None:
            return self._sim.total_steps
        return self._total_steps

    @property
    def time_index(self) -> List[str]:
        if self._backend == "junction_sim" and self._sim is not None:
            return self._sim.time_index
        return self._time_index

    @property
    def approaches(self) -> Dict[str, ApproachState]:
        if self._backend == "junction_sim" and self._sim is not None:
            return self._sim.approaches
        return self._approaches

    @property
    def _total_throughput_pcu(self) -> float:
        if self._backend == "junction_sim" and self._sim is not None:
            return self._sim._total_throughput_pcu
        return self._sumo_throughput_pcu

    # ──────────────────────────────────────────────────────────
    # Unified Step Method
    # ──────────────────────────────────────────────────────────

    def step(self, decision: ControllerDecision) -> StepObservation:
        """Advance simulation by 1 second under the given decision."""
        if self._backend == "junction_sim":
            return self._sim.step(decision)
        return self._step_sumo(decision)

    def _step_sumo(self, decision: ControllerDecision) -> StepObservation:
        """Step SUMO via TraCI and collect empirical microscopic metrics."""
        if self._traci is None:
            raise RuntimeError("TraCI connection is not active.")

        # 1. Apply signal state to SUMO TLS J_center
        tls_state = TLS_STATES["ALL_RED"]
        if decision.signal_state == SignalState.GREEN:
            tls_state = TLS_STATES.get((decision.active_phase_id, SignalState.GREEN), TLS_STATES["ALL_RED"])
        elif decision.signal_state == SignalState.YELLOW:
            tls_state = TLS_STATES.get((decision.active_phase_id, SignalState.YELLOW), TLS_STATES["ALL_RED"])
        elif decision.signal_state == SignalState.ALL_RED:
            tls_state = TLS_STATES["ALL_RED"]

        try:
            self._traci.trafficlight.setRedYellowGreenState(TLS_ID, tls_state)
            self._traci.simulationStep()
        except Exception as exc:
            logger.error(f"TraCI communication error: {exc}. Falling back to junction simulator.")
            self._backend = "junction_sim"
            self._init_junction_sim()
            return self._sim.step(decision)

        current_time_str = (
            self._time_index[self._step_count]
            if self._step_count < len(self._time_index)
            else "10:00:00"
        )

        # 2. Extract microscopic lane measurements for each approach
        total_active_queue_pcu = 0.0
        total_delay = 0.0

        for app_id, lanes in APPROACH_SUMO_LANES.items():
            app = self._approaches[app_id]
            is_green = (
                decision.signal_state == SignalState.GREEN
                and (
                    (decision.active_phase_id == 1 and app_id in ("North_Approach", "South_Approach"))
                    or (decision.active_phase_id == 2 and app_id in ("East_Approach", "West_Approach"))
                )
            )

            halting_vehicles = 0
            waiting_time_sec = 0.0
            for lane_id in lanes:
                try:
                    halting_vehicles += self._traci.lane.getLastStepHaltingNumber(lane_id)
                    waiting_time_sec += self._traci.lane.getWaitingTime(lane_id)
                except Exception:
                    pass

            queue_pcu = round(halting_vehicles * 1.0, 2)
            avg_wait = round(waiting_time_sec / max(1, halting_vehicles), 2)

            if is_green:
                app.time_since_last_served_sec = 0.0
                app.current_signal = SignalState.GREEN
            elif decision.signal_state == SignalState.YELLOW:
                app.current_signal = SignalState.YELLOW
                app.time_since_last_served_sec += 1.0
            else:
                app.current_signal = SignalState.RED
                app.time_since_last_served_sec += 1.0

            app.estimated_queue_pcu = queue_pcu
            app.estimated_queue_vehicles = halting_vehicles
            app.average_wait_time_sec = avg_wait
            app.max_wait_time_sec = max(app.max_wait_time_sec, avg_wait)
            app.is_starving = app.time_since_last_served_sec >= STARVATION_THRESHOLD_SEC

            total_active_queue_pcu += queue_pcu
            total_delay += avg_wait

        try:
            arrived = self._traci.simulation.getArrivedNumber()
            self._sumo_throughput_pcu += arrived * 1.0
        except Exception:
            pass

        avg_delay = total_delay / max(1, len(self._approaches))

        obs = StepObservation(
            step=self._step_count,
            timestamp_str=current_time_str,
            controller_decision=decision,
            approaches={k: v.model_copy() for k, v in self._approaches.items()},
            total_active_queue_pcu=round(total_active_queue_pcu, 2),
            total_throughput_pcu=round(self._total_throughput_pcu, 2),
            average_delay_sec=round(avg_delay, 2),
        )

        self.log.append(obs)
        self._step_count += 1
        return obs

    # ──────────────────────────────────────────────────────────
    # Reset & Lifecycle
    # ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset the simulation environment."""
        if self._backend == "sumo":
            self.close()
            try:
                self._init_sumo()
            except Exception as e:
                logger.warning(f"Failed to reset SUMO ({e}); falling back to junction simulator.")
                self._backend = "junction_sim"
                self._init_junction_sim()
        else:
            self._init_junction_sim()

    def close(self):
        """Clean up simulator and TraCI connections cleanly."""
        if self._traci is not None:
            try:
                self._traci.close()
            except Exception:
                pass
            self._traci = None

        if self._sim is not None:
            self._sim = None
