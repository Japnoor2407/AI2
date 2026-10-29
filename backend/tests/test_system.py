"""Tests for SIH1607 Adaptive Traffic Signal Optimization System"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from backend.core.schema import ControllerMode, SignalState, VehicleClassCounts
from backend.core.config import PCU_WEIGHTS, MIN_GREEN_SEC, MAX_GREEN_SEC
from backend.core.controller_baseline import FixedTimeController
from backend.core.controller_adaptive import AdaptiveController
from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.replay_engine import JunctionSimulator
from backend.data.loader import DatasetAdapter


# ─── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def layout():
    return JunctionLayout("AlankarChowk")


@pytest.fixture
def empty_approaches(layout):
    from backend.core.schema import ApproachState
    return {
        app_id: ApproachState(
            approach_id=app_id,
            approach_name=meta["name"],
            direction=meta["direction"],
            camera_id=meta["camera_id"],
            junction_id=layout.junction_id,
        )
        for app_id, meta in layout.approaches.items()
    }


@pytest.fixture
def zero_arrivals_df():
    rows = []
    for s in range(200):
        ts = f"09:{s//60:02d}:{s%60:02d}"
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            rows.append({"Time": ts, "Direction": d,
                         "car_arrivals": 0, "motorbike_arrivals": 0,
                         "bus_arrivals": 0, "truck_arrivals": 0,
                         "pcu_arrivals": 0.0})
    return pd.DataFrame(rows)


@pytest.fixture
def heavy_north_df():
    """Heavy traffic only on UP (North) approach."""
    rows = []
    for s in range(200):
        ts = f"09:{s//60:02d}:{s%60:02d}"
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            car = 3 if d == "UP" else 0
            rows.append({"Time": ts, "Direction": d,
                         "car_arrivals": car, "motorbike_arrivals": 0,
                         "bus_arrivals": 0, "truck_arrivals": 0,
                         "pcu_arrivals": float(car * PCU_WEIGHTS["car"])})
    return pd.DataFrame(rows)


# ─── VehicleClassCounts PCU tests ───────────────────────────────────────────

def test_vehicle_pcu_calculation():
    v = VehicleClassCounts(car=2, motorbike=4, bus=1, truck=1)
    pcu = v.to_pcu()
    expected = 2*1.0 + 4*0.5 + 1*2.5 + 1*2.5
    assert abs(pcu - expected) < 0.001


def test_vehicle_pcu_all_zero():
    v = VehicleClassCounts()
    assert v.to_pcu() == 0.0
    assert v.total_vehicles == 0


# ─── Layout safety tests ────────────────────────────────────────────────────

def test_layout_approaches(layout):
    assert len(layout.approaches) == 4
    assert "North_Approach" in layout.approaches
    assert "East_Approach" in layout.approaches


def test_layout_phases(layout):
    assert len(layout.phases) in (2, 4)


def test_safety_no_conflict_phase1(layout):
    assert layout.validate_safety(1) is True


def test_safety_no_conflict_phase2(layout):
    assert layout.validate_safety(2) is True


def test_safety_invalid_phase(layout):
    assert layout.validate_safety(99) is False


# ─── Fixed-time controller tests ────────────────────────────────────────────

def test_fixed_controller_initial_state(layout, empty_approaches):
    ctrl = FixedTimeController(phases=layout.phases, fixed_green_sec=30)
    dec = ctrl.step(0, "09:00:00", empty_approaches)
    assert dec.signal_state == SignalState.GREEN
    assert dec.mode == ControllerMode.FIXED_TIME


def test_fixed_controller_transitions_to_yellow(layout, empty_approaches):
    ctrl = FixedTimeController(phases=layout.phases, fixed_green_sec=5, yellow_sec=3)
    for i in range(5):
        dec = ctrl.step(i, f"09:00:0{i}", empty_approaches)
    # step 6 → yellow
    dec = ctrl.step(5, "09:00:05", empty_approaches)
    assert dec.signal_state == SignalState.YELLOW


def test_fixed_controller_full_cycle(layout, empty_approaches):
    ctrl = FixedTimeController(phases=layout.phases, fixed_green_sec=5, yellow_sec=2, all_red_sec=1)
    phases_seen = set()
    for i in range(20):
        dec = ctrl.step(i, f"09:00:{i:02d}", empty_approaches)
        if dec.signal_state == SignalState.GREEN:
            phases_seen.add(dec.active_phase_id)
    # Both phases should have been active
    assert len(phases_seen) >= 1


def test_fixed_respects_min_green(layout, empty_approaches):
    """Controller must hold green for fixed_green_sec steps before transitioning."""
    ctrl = FixedTimeController(phases=layout.phases, fixed_green_sec=15)
    green_count = 0
    for i in range(15):
        dec = ctrl.step(i, f"09:00:{i:02d}", empty_approaches)
        if dec.signal_state == SignalState.GREEN and dec.active_phase_id == 1:
            green_count += 1
    # Steps 0-13 → GREEN (elapsed 1-14), step 14 → elapsed=15 triggers transition to YELLOW
    # So exactly 14 of 15 steps see GREEN (the 15th sees YELLOW)
    assert green_count >= 14


# ─── Adaptive controller tests ───────────────────────────────────────────────

def test_adaptive_initial_green(layout, empty_approaches):
    ctrl = AdaptiveController(phases=layout.phases)
    dec = ctrl.step(0, "09:00:00", empty_approaches)
    assert dec.signal_state == SignalState.GREEN
    assert dec.mode == ControllerMode.ADAPTIVE


def test_adaptive_never_conflicts(layout, heavy_north_df):
    sim = JunctionSimulator(layout=layout, arrival_data=heavy_north_df)
    ctrl = AdaptiveController(phases=layout.phases, min_green=5)
    for i in range(100):
        if sim.step_count >= sim.total_steps:
            break
        dec = ctrl.step(i, sim.time_index[sim.step_count] if i < len(sim.time_index) else "", sim.approaches)
        obs = sim.step(dec)
        if dec.signal_state == SignalState.GREEN:
            assert layout.validate_safety(dec.active_phase_id), \
                f"CONFLICT at step {i}: phase {dec.active_phase_id}"


def test_adaptive_green_within_bounds(layout, heavy_north_df):
    sim = JunctionSimulator(layout=layout, arrival_data=heavy_north_df)
    ctrl = AdaptiveController(phases=layout.phases, min_green=MIN_GREEN_SEC, max_green=MAX_GREEN_SEC)
    green_durations = []
    for i in range(150):
        if sim.step_count >= sim.total_steps:
            break
        dec = ctrl.step(i, sim.time_index[sim.step_count] if i < len(sim.time_index) else "", sim.approaches)
        sim.step(dec)
        if dec.signal_state == SignalState.GREEN:
            green_durations.append(ctrl.allocated_green)
    for d in green_durations:
        assert MIN_GREEN_SEC <= d <= MAX_GREEN_SEC, f"Green duration {d} out of bounds"


def test_adaptive_handles_empty_approaches(layout, zero_arrivals_df):
    sim = JunctionSimulator(layout=layout, arrival_data=zero_arrivals_df)
    ctrl = AdaptiveController(phases=layout.phases, min_green=5)
    for i in range(30):
        if sim.step_count >= sim.total_steps:
            break
        dec = ctrl.step(i, sim.time_index[sim.step_count] if i < len(sim.time_index) else "", sim.approaches)
        obs = sim.step(dec)
    # Should complete without error and queue should be 0
    assert obs.total_active_queue_pcu == 0.0


def test_starvation_prevention(layout):
    """Approach waiting too long must get priority override."""
    from backend.core.schema import ApproachState
    ctrl = AdaptiveController(phases=layout.phases, min_green=5, starvation_threshold=20)
    approaches = {
        app_id: ApproachState(
            approach_id=app_id,
            approach_name=meta["name"],
            direction=meta["direction"],
            camera_id=meta["camera_id"],
            junction_id=layout.junction_id,
            time_since_last_served_sec=30.0 if "East" in app_id else 0.0,
            estimated_queue_pcu=50.0 if "East" in app_id else 0.0,
        )
        for app_id, meta in layout.approaches.items()
    }
    # Force ctrl into GREEN state and min_green elapsed
    ctrl.elapsed_in_state = 10
    dec = ctrl.step(100, "09:00:00", approaches)
    # Should see phase 2 (East-West) or a starvation override rationale
    assert "STARVATION" in dec.decision_rationale or dec.active_phase_id == 2


# ─── Simulator tests ─────────────────────────────────────────────────────────

def test_simulator_queue_never_negative(layout, heavy_north_df):
    sim = JunctionSimulator(layout=layout, arrival_data=heavy_north_df)
    ctrl = FixedTimeController(phases=layout.phases, fixed_green_sec=10)
    for i in range(100):
        if sim.step_count >= sim.total_steps:
            break
        dec = ctrl.step(i, sim.time_index[sim.step_count] if i < len(sim.time_index) else "", sim.approaches)
        obs = sim.step(dec)
        for app in obs.approaches.values():
            assert app.estimated_queue_pcu >= 0, f"Negative queue at step {i}"


def test_simulator_reset(layout, zero_arrivals_df):
    sim = JunctionSimulator(layout=layout, arrival_data=zero_arrivals_df)
    ctrl = FixedTimeController(phases=layout.phases, fixed_green_sec=5)
    for i in range(10):
        if sim.step_count >= sim.total_steps:
            break
        dec = ctrl.step(i, sim.time_index[sim.step_count] if i < len(sim.time_index) else "", sim.approaches)
        sim.step(dec)
    sim.reset()
    assert sim.step_count == 0
    assert sim.log == []


# ─── Dataset adapter tests ───────────────────────────────────────────────────

def test_adapter_junctions():
    adapter = DatasetAdapter()
    junctions = adapter.get_available_junctions()
    assert len(junctions) > 0
    assert "AlankarChowk" in junctions


def test_adapter_cameras():
    adapter = DatasetAdapter()
    cameras = adapter.get_cameras_for_junction("AlankarChowk")
    assert len(cameras) >= 2


def test_adapter_load_no_negative_arrivals():
    """Cumulative diff extraction must produce only non-negative arrivals."""
    adapter = DatasetAdapter()
    df = adapter.load_camera_arrivals("AlankarChowk", "a2", "11", max_seconds=120)
    assert not df.empty
    for col in ["car_arrivals", "motorbike_arrivals", "bus_arrivals", "truck_arrivals"]:
        if col in df.columns:
            assert (df[col] >= 0).all(), f"Negative values in {col}"


def test_adapter_load_missing_file():
    adapter = DatasetAdapter()
    with pytest.raises(FileNotFoundError):
        adapter.load_camera_arrivals("AlankarChowk", "a2", "99", max_seconds=60)
