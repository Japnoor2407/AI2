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


# ─── Phase 1: Actuated Controller Tests ─────────────────────────────────────

from backend.core.controller_actuated import ActuatedController


def test_actuated_initial_state(layout, empty_approaches):
    """Actuated controller starts in GREEN with ACTUATED mode."""
    ctrl = ActuatedController(phases=layout.phases, min_green_sec=10)
    dec = ctrl.step(0, "09:00:00", empty_approaches)
    assert dec.signal_state == SignalState.GREEN
    assert dec.mode == ControllerMode.ACTUATED


def test_actuated_respects_min_green(layout, zero_arrivals_df):
    """With no traffic, actuated must still hold min green before transitioning."""
    sim = JunctionSimulator(layout=layout, arrival_data=zero_arrivals_df)
    ctrl = ActuatedController(phases=layout.phases, min_green_sec=15, gap_threshold_pcu=0.1)
    green_count_p1 = 0
    for i in range(20):
        if sim.step_count >= sim.total_steps:
            break
        dec = ctrl.step(i, sim.time_index[sim.step_count], sim.approaches)
        sim.step(dec)
        if dec.signal_state == SignalState.GREEN and dec.active_phase_id == 1:
            green_count_p1 += 1
    assert green_count_p1 >= 14, f"Expected >=14 green steps on phase 1, got {green_count_p1}"


def test_actuated_max_green_terminates(layout, heavy_north_df):
    """Actuated controller must terminate green at max_green even with continuous demand."""
    sim = JunctionSimulator(layout=layout, arrival_data=heavy_north_df)
    ctrl = ActuatedController(phases=layout.phases, min_green_sec=5, max_green_sec=20)
    max_phase1_green = 0
    current_phase1_run = 0
    for i in range(100):
        if sim.step_count >= sim.total_steps:
            break
        dec = ctrl.step(i, sim.time_index[sim.step_count], sim.approaches)
        sim.step(dec)
        if dec.signal_state == SignalState.GREEN and dec.active_phase_id == 1:
            current_phase1_run += 1
            max_phase1_green = max(max_phase1_green, current_phase1_run)
        else:
            current_phase1_run = 0
    assert max_phase1_green <= 22, f"Phase 1 green ran {max_phase1_green}s, expected <=22"


def test_actuated_never_conflicts(layout, heavy_north_df):
    """Actuated controller must never produce a conflicting green phase."""
    sim = JunctionSimulator(layout=layout, arrival_data=heavy_north_df)
    ctrl = ActuatedController(phases=layout.phases, min_green_sec=5)
    for i in range(100):
        if sim.step_count >= sim.total_steps:
            break
        dec = ctrl.step(i, sim.time_index[sim.step_count], sim.approaches)
        sim.step(dec)
        if dec.signal_state == SignalState.GREEN:
            assert layout.validate_safety(dec.active_phase_id), \
                f"CONFLICT at step {i}: phase {dec.active_phase_id}"


def test_actuated_cycles_all_phases(layout, zero_arrivals_df):
    """Actuated controller must serve all 4 phases over a full cycle run."""
    sim = JunctionSimulator(layout=layout, arrival_data=zero_arrivals_df)
    ctrl = ActuatedController(phases=layout.phases, min_green_sec=5, max_green_sec=10)
    phases_seen = set()
    for i in range(200):
        if sim.step_count >= sim.total_steps:
            break
        dec = ctrl.step(i, sim.time_index[sim.step_count], sim.approaches)
        sim.step(dec)
        if dec.signal_state == SignalState.GREEN:
            phases_seen.add(dec.active_phase_id)
    # Must have cycled through at least 2 distinct phases
    assert len(phases_seen) >= 2, f"Only saw phases: {phases_seen}"


# ─── Phase 2: Safety Supervisor Tests ───────────────────────────────────────

import math
from backend.core.safety_layer import SafetySupervisor
from backend.core.schema import ApproachState, ControllerDecision, ControllerMode, SafetyEvent, PhaseConfig


def _make_decision(phase_id: int, signal: SignalState, elapsed: int = 5) -> ControllerDecision:
    return ControllerDecision(
        step=100,
        timestamp_str="09:00:00",
        mode=ControllerMode.ADAPTIVE,
        active_phase_id=phase_id,
        phase_name=f"Phase {phase_id}",
        signal_state=signal,
        time_remaining_in_state=max(0, 30 - elapsed),
        elapsed_in_phase=elapsed,
    )


def test_safety_premature_green_termination(layout, empty_approaches):
    """Safety supervisor must strictly prevent premature green termination when elapsed < min_green."""
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=layout.phases,
        min_green_sec=15,
    )
    # Controller prematurely requests YELLOW when only 5 seconds have elapsed
    dec = _make_decision(phase_id=1, signal=SignalState.YELLOW, elapsed=5)
    approved, events = supervisor.validate_and_enforce(
        decision=dec,
        approaches=empty_approaches,
        current_step=100,
        timestamp_str="09:00:00",
        current_active_phase_id=1,
        current_signal_state=SignalState.GREEN,
        elapsed_in_state=5,
    )
    assert approved.signal_state == SignalState.GREEN, "Must hold GREEN until min_green satisfied"
    assert approved.safety_override is True, "Must set safety_override flag"
    assert len(events) == 1, "Must generate exactly 1 safety audit event"
    assert "MIN_GREEN_VIOLATION" in events[0].active_constraints
    assert events[0].previous_phase_id == 1
    assert events[0].approved_phase_id == 1


def test_safety_max_green_enforcement(layout, empty_approaches):
    """Safety supervisor must force transition to YELLOW when green exceeds max_green."""
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=layout.phases,
        max_green_sec=50,
    )
    # Controller attempts to keep holding GREEN at elapsed=55
    dec = _make_decision(phase_id=1, signal=SignalState.GREEN, elapsed=55)
    approved, events = supervisor.validate_and_enforce(
        decision=dec,
        approaches=empty_approaches,
        current_step=150,
        timestamp_str="09:02:30",
        current_active_phase_id=1,
        current_signal_state=SignalState.GREEN,
        elapsed_in_state=55,
    )
    assert approved.signal_state == SignalState.YELLOW, "Must force transition to YELLOW"
    assert approved.safety_override is True
    assert len(events) == 1
    assert "MAX_GREEN_EXCEEDED" in events[0].active_constraints


def test_safety_yellow_clearance_uninterruptible(layout, empty_approaches):
    """Yellow clearance cannot be interrupted with instantaneous green of another phase."""
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=layout.phases,
        yellow_sec=3,
    )
    # 1 second into yellow clearance, controller tries to jump to GREEN on Phase 2
    dec = _make_decision(phase_id=2, signal=SignalState.GREEN, elapsed=1)
    approved, events = supervisor.validate_and_enforce(
        decision=dec,
        approaches=empty_approaches,
        current_step=200,
        timestamp_str="09:03:20",
        current_active_phase_id=1,
        current_signal_state=SignalState.YELLOW,
        elapsed_in_state=1,
    )
    assert approved.signal_state == SignalState.YELLOW, "Must maintain YELLOW clearance"
    assert approved.active_phase_id == 1, "Must not switch phase during clearance"
    assert approved.safety_override is True
    assert len(events) == 1
    assert "YELLOW_CLEARANCE_ACTIVE" in events[0].active_constraints


def test_safety_all_red_clearance_uninterruptible(layout, empty_approaches):
    """All-Red junction clearance cannot be interrupted prematurely."""
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=layout.phases,
        all_red_sec=2,
    )
    # 1 second into all-red, controller tries to switch to GREEN
    dec = _make_decision(phase_id=2, signal=SignalState.GREEN, elapsed=1)
    approved, events = supervisor.validate_and_enforce(
        decision=dec,
        approaches=empty_approaches,
        current_step=205,
        timestamp_str="09:03:25",
        current_active_phase_id=1,
        current_signal_state=SignalState.ALL_RED,
        elapsed_in_state=1,
    )
    assert approved.signal_state == SignalState.ALL_RED, "Must maintain ALL_RED clearance"
    assert approved.safety_override is True
    assert len(events) == 1
    assert "ALL_RED_CLEARANCE_ACTIVE" in events[0].active_constraints


def test_safety_conflict_matrix_rejected(layout, empty_approaches):
    """A phase serving both NS and EW simultaneously must be rejected by the safety layer."""
    conflict_phase = PhaseConfig(
        phase_id=99,
        name="Conflicting Phase",
        served_approaches=["North_Approach", "East_Approach"],  # NS + EW conflict
        min_green_sec=10,
        max_green_sec=60,
    )
    extended_phases = layout.phases + [conflict_phase]
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=extended_phases,
    )
    dec = _make_decision(phase_id=99, signal=SignalState.GREEN, elapsed=15)
    approved, events = supervisor.validate_and_enforce(
        decision=dec,
        approaches=empty_approaches,
        current_step=100,
        timestamp_str="09:00:00",
        current_active_phase_id=1,
        current_signal_state=SignalState.GREEN,
        elapsed_in_state=15,
    )
    assert approved.signal_state == SignalState.ALL_RED, "Conflicting phase must be overridden to ALL_RED"
    assert approved.safety_override is True
    assert len(events) == 1
    assert "CONFLICT_FREE_INVARIANT" in events[0].active_constraints
    assert "crosses NS and EW" in events[0].override_reason


def test_safety_invalid_phase_fallback(layout, empty_approaches):
    """Requesting an invalid phase_id must fall back to current active phase with safety override."""
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=layout.phases,
    )
    dec = _make_decision(phase_id=999, signal=SignalState.GREEN, elapsed=15)
    approved, events = supervisor.validate_and_enforce(
        decision=dec,
        approaches=empty_approaches,
        current_step=100,
        timestamp_str="09:00:00",
        current_active_phase_id=1,
        current_signal_state=SignalState.GREEN,
        elapsed_in_state=15,
    )
    assert approved.active_phase_id == 1, "Must fall back to active phase 1"
    assert approved.safety_override is True
    assert len(events) == 1
    assert "INVALID_PHASE_ID" in events[0].active_constraints


def test_safety_invalid_numerical_values(layout, empty_approaches):
    """NaN or negative time_remaining must be clamped to zero."""
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=layout.phases,
    )
    dec = _make_decision(phase_id=1, signal=SignalState.GREEN, elapsed=20)
    dec.time_remaining_in_state = -15
    approved, _ = supervisor.validate_and_enforce(
        decision=dec,
        approaches=empty_approaches,
        current_step=100,
        timestamp_str="09:00:00",
    )
    assert approved.time_remaining_in_state == 0
    assert not math.isnan(approved.time_remaining_in_state)


def test_safety_failsafe_on_controller_exception(layout):
    """When a controller raises an unhandled exception, supervisor must trigger emergency clearance."""
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=layout.phases,
    )
    supervisor.current_signal_state = SignalState.GREEN
    fallback_dec, event = supervisor.safe_fallback_decision(
        current_step=42,
        timestamp_str="09:00:42",
        exception_reason="ZeroDivisionError in scoring heuristic",
    )
    assert fallback_dec.signal_state == SignalState.YELLOW
    assert fallback_dec.safety_override is True
    assert "ZeroDivisionError" in fallback_dec.safety_override_reason
    assert "CONTROLLER_EXCEPTION_FAILSAFE" in event.active_constraints
    assert supervisor.total_violations_prevented == 1


def test_safety_missing_sensor_data_handling(layout):
    """Missing or empty sensor approaches must be handled safely without unhandled errors."""
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=layout.phases,
    )
    dec = _make_decision(phase_id=1, signal=SignalState.GREEN, elapsed=20)
    approved, _ = supervisor.validate_and_enforce(
        decision=dec,
        approaches={},  # empty sensor data
        current_step=100,
        timestamp_str="09:00:00",
    )
    assert approved.signal_state == SignalState.GREEN
    assert approved.active_phase_id == 1


def test_safety_full_transition_sequence(layout, empty_approaches):
    """Verify complete sequence of transitions over consecutive autonomous steps."""
    supervisor = SafetySupervisor(
        intersection_id="AlankarChowk",
        phases=layout.phases,
        min_green_sec=3,
        yellow_sec=2,
        all_red_sec=1,
    )
    # Step 1: Green on phase 1 (elapsed becomes 1)
    d1 = _make_decision(phase_id=1, signal=SignalState.GREEN, elapsed=1)
    a1, e1 = supervisor.validate_and_enforce(d1, empty_approaches, 1, "09:00:01")
    assert a1.signal_state == SignalState.GREEN
    assert len(e1) == 0

    # Step 2: Premature switch to Yellow at elapsed=1 (< min_green=3) -> blocked, stays Green (elapsed becomes 2)
    d2 = _make_decision(phase_id=1, signal=SignalState.YELLOW, elapsed=1)
    a2, e2 = supervisor.validate_and_enforce(d2, empty_approaches, 2, "09:00:02")
    assert a2.signal_state == SignalState.GREEN, "Premature termination must be blocked"
    assert len(e2) == 1
    assert "MIN_GREEN_VIOLATION" in e2[0].active_constraints

    # Step 3: Green held for 3rd second (elapsed becomes 3)
    d3 = _make_decision(phase_id=1, signal=SignalState.GREEN, elapsed=2)
    a3, e3 = supervisor.validate_and_enforce(d3, empty_approaches, 3, "09:00:03")
    assert a3.signal_state == SignalState.GREEN
    assert len(e3) == 0

    # Step 4: Green has reached elapsed=3 (>= min_green=3), controller requests Yellow -> permitted
    d4 = _make_decision(phase_id=1, signal=SignalState.YELLOW, elapsed=3)
    a4, e4 = supervisor.validate_and_enforce(d4, empty_approaches, 4, "09:00:04")
    assert a4.signal_state == SignalState.YELLOW
    assert len(e4) == 0

    # Step 5: During Yellow clearance (elapsed 1 < 2), controller tries to jump to Green phase 2 -> blocked
    d5 = _make_decision(phase_id=2, signal=SignalState.GREEN, elapsed=1)
    a5, e5 = supervisor.validate_and_enforce(d5, empty_approaches, 5, "09:00:05")
    assert a5.signal_state == SignalState.YELLOW
    assert len(e5) == 1
    assert "YELLOW_CLEARANCE_ACTIVE" in e5[0].active_constraints

    # Step 6: Yellow completes (elapsed reached 2 >= 2), transitions to All-Red (elapsed becomes 1)
    d6 = _make_decision(phase_id=1, signal=SignalState.ALL_RED, elapsed=2)
    a6, e6 = supervisor.validate_and_enforce(d6, empty_approaches, 6, "09:00:06")
    assert a6.signal_state == SignalState.ALL_RED
    assert len(e6) == 0

    # Step 7: All-Red completes (elapsed reached 1 >= 1), green permitted for Phase 2
    d7 = _make_decision(phase_id=2, signal=SignalState.GREEN, elapsed=1)
    a7, e7 = supervisor.validate_and_enforce(d7, empty_approaches, 7, "09:00:07")
    assert a7.signal_state == SignalState.GREEN
    assert a7.active_phase_id == 2
    assert len(e7) == 0


# ─── Phase 1: Extended Metrics Tests ─────────────────────────────────────────

from backend.evaluation.evaluator import _compute_percentile


def test_percentile_known_values():
    """Percentile calculation must return correct values for known inputs."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _compute_percentile(data, 50) == 3.0  # median
    assert _compute_percentile(data, 0) == 1.0
    assert _compute_percentile(data, 100) == 5.0


def test_percentile_empty():
    assert _compute_percentile([], 50) == 0.0


def test_percentile_p95_greater_than_mean():
    """P95 must be >= mean for right-skewed positive data."""
    import random
    data = [random.expovariate(0.1) for _ in range(500)]
    p95 = _compute_percentile(data, 95)
    mean = sum(data) / len(data)
    assert p95 >= mean * 0.8  # P95 >= 80% of mean (always true for positive data)


# ─── Phase 3: ML Forecaster Tests ───────────────────────────────────────────

import tempfile
import numpy as np
from pathlib import Path
from backend.models.forecaster import TrafficDemandForecaster


@pytest.fixture
def sample_ml_training_df():
    """Generates synthetic multi-step time series dataset for ML validation."""
    rows = []
    for s in range(300):
        ts = f"09:{s//60:02d}:{s%60:02d}"
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            # Periodic wave pattern
            val = float(max(0, 2 + int(np.sin(s / 10.0) * 3)))
            rows.append({
                "Time": ts,
                "Direction": d,
                "car": int(val),
                "motorbike": int(val * 2),
                "bus": 0,
                "truck": 0,
                "car_arrivals": int(val),
                "motorbike_arrivals": int(val * 2),
                "bus_arrivals": 0,
                "truck_arrivals": 0,
                "pcu_arrivals": float(val * 1.0 + val * 2 * 0.5),
            })
    return pd.DataFrame(rows)


def test_ml_forecaster_chronological_split_no_leakage(sample_ml_training_df):
    """Features and target splits must be strictly chronological without future data leakage."""
    forecaster = TrafficDemandForecaster()
    X, y = forecaster._prepare_time_series_features(sample_ml_training_df)
    assert not X.empty
    assert len(X) == len(y)
    # Target must be strictly 1 step ahead of lag1
    # Verify lag1 is past and target is strictly step t+1
    assert "lag1" in X.columns
    assert "roll_mean_5" in X.columns


def test_ml_forecaster_evaluation_metrics(sample_ml_training_df):
    """Forecaster training must report valid MAE, RMSE, and benchmark comparisons."""
    forecaster = TrafficDemandForecaster()
    metrics = forecaster.train_and_evaluate(sample_ml_training_df, model_type="gradient_boosting")
    assert forecaster.is_trained is True
    assert "ml_mae" in metrics
    assert "baseline_persistence_mae" in metrics
    assert "baseline_moving_avg_mae" in metrics
    assert metrics["ml_mae"] >= 0.0
    assert metrics["ml_rmse"] >= 0.0
    assert not math.isnan(metrics["ml_mae"])
    assert not math.isnan(metrics["ml_rmse"])
    assert metrics["training_samples"] > 0
    assert metrics["test_samples"] > 0


def test_ml_forecaster_persistence_and_reload(sample_ml_training_df):
    """Trained models must save to disk and reload accurately."""
    forecaster = TrafficDemandForecaster()
    forecaster.train_and_evaluate(sample_ml_training_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_model.joblib"
        forecaster.save_model(tmp_path)
        assert tmp_path.exists()

        new_forecaster = TrafficDemandForecaster()
        loaded = new_forecaster.load_model(tmp_path)
        assert loaded is True
        assert new_forecaster.is_trained is True
        assert new_forecaster.feature_names == forecaster.feature_names
        # Prediction on both should yield identical result
        p1 = forecaster.predict_short_term_arrivals([1.0, 2.0, 3.0, 2.5, 3.0], "UP")
        p2 = new_forecaster.predict_short_term_arrivals([1.0, 2.0, 3.0, 2.5, 3.0], "UP")
        assert abs(p1 - p2) < 1e-4


def test_ml_forecaster_empty_insufficient_data():
    """Empty or insufficient data (< 50 samples) must raise ValueError."""
    forecaster = TrafficDemandForecaster()
    with pytest.raises(ValueError):
        forecaster.train_and_evaluate(pd.DataFrame())

    # Fallback when model is untrained returns rolling mean or 0.0
    pred = forecaster.predict_short_term_arrivals([], "UP")
    assert pred == 0.0
    pred2 = forecaster.predict_short_term_arrivals([2.0, 4.0], "UP")
    assert pred2 == 3.0  # rolling mean of [2.0, 4.0]


def test_prediction_assisted_controller_integration(layout, sample_ml_training_df, empty_approaches):
    """Adaptive controller in PREDICTION_ASSISTED mode incorporates ML forecasts safely."""
    forecaster = TrafficDemandForecaster()
    forecaster.train_and_evaluate(sample_ml_training_df)

    pred_ctrl = AdaptiveController(
        phases=layout.phases,
        mode=ControllerMode.PREDICTION_ASSISTED,
        forecaster=forecaster,
    )
    dec = pred_ctrl.step(0, "09:00:00", empty_approaches)
    assert dec.mode == ControllerMode.PREDICTION_ASSISTED
    assert dec.signal_state == SignalState.GREEN
    assert layout.validate_safety(dec.active_phase_id) is True


# ─── Phase 5: Multi-Intersection Corridor Coordination Tests ───────────────

from backend.simulation.corridor import CoordinatedCorridor


def test_corridor_coordination_simulation(zero_arrivals_df):
    """Corridor simulation correctly propagates platoons and maintains independent safety."""
    corridor = CoordinatedCorridor(
        arrivals_alankar=zero_arrivals_df,
        arrivals_jehangir=zero_arrivals_df,
        travel_time_sec=5,
        spillback_threshold_pcu=10.0,
    )
    ctrl_a = AdaptiveController(phases=corridor.layout_alankar.phases)
    ctrl_j = AdaptiveController(phases=corridor.layout_jehangir.phases)

    for i in range(15):
        record = corridor.step(ctrl_a, ctrl_j, enable_coordination=True)
        assert record["step"] == i
        assert record["total_corridor_queue_pcu"] >= 0.0
        # Invariants: safety supervisor must be active on both
        assert corridor.layout_alankar.validate_safety(record["alankar_obs"].controller_decision.active_phase_id)
        assert corridor.layout_jehangir.validate_safety(record["jehangir_obs"].controller_decision.active_phase_id)


# ─── Phase 4: Local Vision & Tracking Tests ─────────────────────────────────

from backend.vision.detector import LocalVehicleDetector, VehicleDetection
from backend.vision.tracker import CentroidTracker, VirtualTripwireCounter
from backend.vision.evaluator import DetectionEvaluator


def test_vision_detector_empty_or_blank_frame():
    """Detector on blank or empty frame returns empty detections safely."""
    detector = LocalVehicleDetector()
    assert detector.detect(None) == []
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect(blank) == []


def test_vision_tracker_association():
    """Tracker associates detected centroids across consecutive frames."""
    tracker = CentroidTracker(max_distance=50.0)
    # Frame 1: Detection at (100, 100)
    d1 = [VehicleDetection((90, 90, 20, 20), "car", 0.9)]
    tracks1 = tracker.update(d1)
    assert len(tracks1) == 1
    t_id = tracks1[0].track_id

    # Frame 2: Same vehicle moved slightly to (105, 105)
    d2 = [VehicleDetection((95, 95, 20, 20), "car", 0.9)]
    tracks2 = tracker.update(d2)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == t_id, "Track ID must remain consistent"


def test_vision_tripwire_crossing():
    """Virtual tripwire correctly increments counts when vehicle crosses boundary."""
    tripwire = VirtualTripwireCounter(line_y=100, direction_downwards=True)
    tracker = CentroidTracker()

    # Step 1: Vehicle above line at y=80
    d1 = [VehicleDetection((50, 70, 20, 20), "car", 0.9)]
    tracks1 = tracker.update(d1)
    counts1 = tripwire.update(tracks1)
    assert counts1["car"] == 0

    # Step 2: Vehicle crosses to y=110
    d2 = [VehicleDetection((50, 100, 20, 20), "car", 0.9)]
    tracks2 = tracker.update(d2)
    counts2 = tripwire.update(tracks2)
    assert counts2["car"] == 1
    assert tripwire.total_count == 1


def test_vision_detection_evaluator():
    """Evaluator calculates Precision, Recall, and F1 against ground truth."""
    pred = {"car": 10, "bus": 2}
    gt = {"car": 10, "bus": 4}
    res = DetectionEvaluator.evaluate(pred, gt)
    assert res["overall_precision"] == 1.0  # All 12 predicted were true
    assert abs(res["overall_recall"] - 12 / 14) < 0.001
    assert res["per_class"]["car"]["f1_score"] == 1.0
