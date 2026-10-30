"""
Comprehensive unit and integration tests for Emergency Vehicle Priority (EVP)
and Strategy Comparison Lab.
"""

import pytest

from backend.core.emergency import (
    EmergencyPriorityManager,
    EmergencyVehicleType,
    EmergencyPriorityStatus,
)
from backend.core.schema import (
    PhaseConfig,
    ControllerDecision,
    SignalState,
    ControllerMode,
)
from backend.evaluation.comparison_lab import (
    run_comparison_lab,
    get_lab_history,
    get_lab_report,
    export_lab_csv,
)


@pytest.fixture
def sample_phases():
    return [
        PhaseConfig(phase_id=1, name="North-South Green", served_approaches=["N", "S"], min_green_sec=10, max_green_sec=55),
        PhaseConfig(phase_id=2, name="East-West Green", served_approaches=["E", "W"], min_green_sec=10, max_green_sec=55),
    ]


class TestEmergencyPriorityManager:
    def test_request_and_clear_priority(self, sample_phases):
        mgr = EmergencyPriorityManager(phases=sample_phases, yellow_sec=3, all_red_sec=2)
        event = mgr.request_priority(
            vehicle_type=EmergencyVehicleType.AMBULANCE,
            approach_id="E",
            current_step=10,
        )
        assert event.vehicle_type == EmergencyVehicleType.AMBULANCE
        assert event.target_phase_id == 2  # E is served by Phase 2
        assert mgr.active_event is not None

        # Clear priority
        mgr.clear_active_priority(current_step=25)
        assert mgr.active_event is None
        assert len(mgr.event_history) == 1
        assert mgr.event_history[0].clearance_latency_sec == 15.0
        assert mgr.event_history[0].status == EmergencyPriorityStatus.CLEARED

    def test_intercept_decision_clearance_stages(self, sample_phases):
        mgr = EmergencyPriorityManager(phases=sample_phases, yellow_sec=2, all_red_sec=1)
        mgr.request_priority(
            vehicle_type=EmergencyVehicleType.FIRE_TRUCK,
            approach_id="W",
            current_step=5,
        )

        normal_decision = ControllerDecision(
            step=5,
            timestamp_str="08:00:05",
            mode=ControllerMode.ADAPTIVE,
            active_phase_id=1,  # NS is currently active
            phase_name="North-South Green",
            signal_state=SignalState.GREEN,
            time_remaining_in_state=20,
            elapsed_in_phase=5,
            phase_demand_scores={"1": 10.0, "2": 5.0},
            decision_rationale="Normal green",
        )

        # Stage 1: Yellow transition
        d1 = mgr.intercept_decision(normal_decision, 5, "08:00:05")
        assert d1.signal_state == SignalState.YELLOW
        assert "Emergency yellow" in d1.decision_rationale

        # Stage 2: Second step of yellow
        d2 = mgr.intercept_decision(normal_decision, 6, "08:00:06")
        assert d2.signal_state == SignalState.YELLOW

        # Stage 3: All-red clearance
        d3 = mgr.intercept_decision(normal_decision, 7, "08:00:07")
        assert d3.signal_state == SignalState.ALL_RED

        # Stage 4: Priority Green granted to Phase 2 (serving W)
        d4 = mgr.intercept_decision(normal_decision, 8, "08:00:08")
        assert d4.signal_state == SignalState.GREEN
        assert d4.active_phase_id == 2


class TestEmergencyAPI:
    def test_emergency_lifecycle_api(self):
        from backend.core.state import app_state
        from backend.simulation.junction_layout import JunctionLayout
        from backend.main import (
            emergency_request,
            emergency_status,
            emergency_clear,
            emergency_events,
            EmergencyRequest,
        )
        if not app_state._initialized:
            app_state.layout = JunctionLayout("AlankarChowk")
            app_state.emergency_manager = EmergencyPriorityManager(phases=app_state.layout.phases)
            app_state._initialized = True

        # 1. Request priority
        res = emergency_request(EmergencyRequest(vehicle_type="AMBULANCE", approach_id="N"))
        assert res["status"] == "ok"
        assert res["event"]["vehicle_type"] == "AMBULANCE"

        # 2. Check status
        st = emergency_status()
        assert st["is_active"] is True

        # 3. Clear priority
        clr = emergency_clear()
        assert clr["status"] == "ok"

        # 4. Status should now be inactive
        st_after = emergency_status()
        assert st_after["is_active"] is False

        # 5. History should record the event
        hist = emergency_events()
        assert len(hist["events"]) >= 1


class TestComparisonLab:
    def test_run_comparison_lab_all_strategies(self):
        # Run a short 30s benchmark to test execution and ranking logic quickly
        report = run_comparison_lab(
            junction_id="AlankarChowk",
            date="11",
            duration_sec=30,
            selected_strategies=["fixed_time", "actuated", "adaptive", "max_pressure", "deep_rl"],
            seed=42,
        )

        assert report.session_id is not None
        assert "fixed_time" in report.results
        assert "deep_rl" in report.results
        assert "max_pressure" in report.results
        assert len(report.rankings) == 5

        # Check rankings order (composite score descending)
        scores = [r.composite_score for r in report.rankings]
        assert scores == sorted(scores, reverse=True)

        # Check that fixed_time baseline metrics exist
        ft_metric = report.results["fixed_time"]
        assert ft_metric.mean_queue_pcu >= 0.0

        # Check CSV export
        csv_data = export_lab_csv(report.session_id)
        assert "Rank,Strategy" in csv_data
        assert "Fixed-Time" in csv_data

    def test_comparison_lab_api(self):
        from backend.main import (
            lab_run,
            lab_report_detail,
            lab_history,
            lab_export,
            LabRunRequest,
        )
        res = lab_run(
            LabRunRequest(
                junction_id="AlankarChowk",
                date="11",
                duration_sec=30,
                selected_strategies=["fixed_time", "adaptive"],
            )
        )
        assert res["status"] == "ok"
        session_id = res["session_id"]

        # Detail endpoint
        det = lab_report_detail(session_id)
        assert det["status"] == "ok"
        assert det["report"]["session_id"] == session_id

        # History endpoint
        hist = lab_history()
        assert hist["status"] == "ok"
        assert len(hist["history"]) >= 1

        # JSON Export
        exp_json = lab_export(session_id, format="json")
        assert exp_json.status_code == 200

        # CSV Export
        exp_csv = lab_export(session_id, format="csv")
        assert exp_csv.status_code == 200
        assert exp_csv.media_type == "text/csv"
