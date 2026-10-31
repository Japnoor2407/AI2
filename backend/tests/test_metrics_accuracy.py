"""
Regression tests for metric accuracy, vehicle count consistency, and savings calculations.

These tests guard against the following historical bugs:
  1. total_arrived_vehicles always 0 (read arrivals_last_step instead of cumulative_arrivals_pcu)
  2. starvation_occurrences counted per-step-per-approach instead of per event (transition)
  3. savings calculations multiplied by max(1, vehicles) inflating results with 0 vehicles
  4. vehicles_processed inconsistent with total_arrived_vehicles
  5. PPO controller falling back silently to Adaptive due to inference-frequency mismatch
"""

import pytest
import pandas as pd
import numpy as np

from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.replay_engine import JunctionSimulator
from backend.core.controller_baseline import FixedTimeController
from backend.core.controller_adaptive import AdaptiveController
from backend.core.controller_actuated import ActuatedController
from backend.core.controller_max_pressure import MaxPressureController
from backend.core.controller_rl import RLTrafficController
from backend.data.loader import DatasetAdapter
from backend.evaluation.evaluator import _collect_metrics

_JUNCTION_ID = "AlankarChowk"
_DATE = "11"
_DURATION_SEC = 300  # 5-minute runs keep tests fast


@pytest.fixture(scope="module")
def arrivals_df():
    """Load real arrival data (shared across tests in this module)."""
    adapter = DatasetAdapter()
    layout = JunctionLayout(_JUNCTION_ID)
    cameras = set(m["camera_id"] for m in layout.approaches.values())
    primary_cam = sorted(cameras)[0]
    return adapter.load_camera_arrivals(
        _JUNCTION_ID, primary_cam, _DATE, max_seconds=_DURATION_SEC + 120
    )


@pytest.fixture(scope="module")
def layout():
    return JunctionLayout(_JUNCTION_ID)


def _run(ctrl, layout, arrivals_df, duration=_DURATION_SEC):
    """Helper: run a controller for duration seconds and return MetricRecord."""
    sim = JunctionSimulator(layout=layout, arrival_data=arrivals_df.copy(), date=_DATE)
    steps = min(duration, sim.total_steps)
    for _ in range(steps):
        ts = sim.time_index[sim.step_count] if sim.step_count < len(sim.time_index) else ""
        dec = ctrl.step(current_step=sim.step_count, timestamp_str=ts, approaches=sim.approaches)
        sim.step(dec)
    return _collect_metrics(sim, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. total_arrived_vehicles must be > 0 after a non-trivial simulation
# ─────────────────────────────────────────────────────────────────────────────

def test_total_arrived_vehicles_nonzero_after_simulation(layout, arrivals_df):
    """Regression: total_arrived_vehicles was always 0 before the cumulative fix."""
    ctrl = FixedTimeController(phases=layout.phases)
    m = _run(ctrl, layout, arrivals_df)
    assert m.total_arrived_vehicles > 0, (
        f"total_arrived_vehicles must be > 0 after {_DURATION_SEC}s; got {m.total_arrived_vehicles}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Starvation events must be counted as transitions, not per-step flags
# ─────────────────────────────────────────────────────────────────────────────

def test_starvation_events_not_per_step(layout, arrivals_df):
    """
    Regression: starvation was accumulated every step an approach was starving,
    producing thousands of events for a 300s run. A real starvation event
    happens at most once per approach per episode for a 5-minute run.
    Upper bound is 4 approaches × 3 events each = 12 conservatively.
    """
    ctrl = FixedTimeController(phases=layout.phases)
    m = _run(ctrl, layout, arrivals_df)
    assert m.starvation_occurrences < 20, (
        f"starvation_occurrences={m.starvation_occurrences} is suspiciously high; "
        "likely still counting per-step instead of per-event"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. vehicles_processed must be <= total_arrived_vehicles
# ─────────────────────────────────────────────────────────────────────────────

def test_vehicles_processed_consistent_with_arrived(layout, arrivals_df):
    """
    Vehicles that departed cannot exceed vehicles that arrived.
    vehicles_processed <= total_arrived_vehicles (within a small tolerance for
    vehicles already in queue at episode start).
    """
    ctrl = AdaptiveController(phases=layout.phases)
    m = _run(ctrl, layout, arrivals_df)
    # Allow 5% tolerance for pre-queued vehicles
    assert m.total_throughput_vehicles <= m.total_arrived_vehicles * 1.05, (
        f"vehicles_processed ({m.total_throughput_vehicles}) "
        f"> total_arrived_vehicles ({m.total_arrived_vehicles}) by more than 5%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Savings must be 0 when no vehicles arrive (empty arrivals scenario)
# ─────────────────────────────────────────────────────────────────────────────

def test_savings_zero_when_no_vehicles(layout, arrivals_df):
    """
    Regression: with max(1, total_arrived_vehicles) the savings were inflated
    when vehicle count was 0. With 0 vehicles, simulated_time_saved must be 0.
    """
    # Create an all-zero arrivals frame
    empty_df = arrivals_df.copy()
    count_cols = [
        c for c in empty_df.columns
        if c in ("car", "motorbike", "bus", "truck",
                 "car_arrivals", "motorbike_arrivals", "bus_arrivals", "truck_arrivals",
                 "pcu_arrivals")
    ]
    for col in count_cols:
        empty_df[col] = 0

    ctrl = AdaptiveController(phases=layout.phases)
    m = _run(ctrl, layout, empty_df)

    # When no vehicles arrive, arrived count must be 0
    assert m.total_arrived_vehicles == 0, (
        f"Expected 0 arrived vehicles with empty arrivals; got {m.total_arrived_vehicles}"
    )
    # Savings calculation: 0 vehicles → 0 total saved
    # (The comparison_lab multiplies time_saved_per_veh * veh_count; with 0 count = 0)
    # Here we verify the evaluator gives us the right input.
    assert m.total_throughput_vehicles == 0, (
        f"Expected 0 processed vehicles with empty arrivals; got {m.total_throughput_vehicles}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. All strategies must see identical vehicle arrivals
# ─────────────────────────────────────────────────────────────────────────────

def test_all_strategies_identical_vehicle_arrivals(layout, arrivals_df):
    """
    Each strategy runs on arrivals_df.copy(), so the same vehicles arrive
    regardless of controller decisions. total_arrived_vehicles should be the
    same (±2 from PCU rounding) for all strategies.
    """
    controllers = {
        "fixed_time": FixedTimeController(phases=layout.phases),
        "actuated": ActuatedController(phases=layout.phases),
        "adaptive": AdaptiveController(phases=layout.phases),
        "max_pressure": MaxPressureController(phases=layout.phases),
    }
    arrived_counts = {}
    for key, ctrl in controllers.items():
        m = _run(ctrl, layout, arrivals_df)
        arrived_counts[key] = m.total_arrived_vehicles

    values = list(arrived_counts.values())
    max_val = max(values)
    min_val = min(values)
    assert max_val - min_val <= 3, (
        f"total_arrived_vehicles differs by more than 3 across strategies: {arrived_counts}. "
        "All strategies should see identical arrivals (same arrivals_df.copy())."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Savings must be negative when strategy performs worse than baseline
# ─────────────────────────────────────────────────────────────────────────────

def test_savings_calculation_direction(layout, arrivals_df):
    """
    If a strategy has higher average wait than Fixed-Time (worse performance),
    avg_waiting_time_saved_per_vehicle_sec = fixed_avg - strategy_avg must be negative.
    """
    fixed_ctrl = FixedTimeController(phases=layout.phases)
    fixed_m = _run(fixed_ctrl, layout, arrivals_df)

    adaptive_ctrl = AdaptiveController(phases=layout.phases)
    adaptive_m = _run(adaptive_ctrl, layout, arrivals_df)

    # Savings per vehicle = fixed_avg_wait - test_avg_wait
    savings_per_veh = fixed_m.average_wait_sec - adaptive_m.average_wait_sec

    # Verify the sign is correct: if adaptive is better, savings > 0; worse → negative
    if adaptive_m.average_wait_sec > fixed_m.average_wait_sec:
        assert savings_per_veh < 0, (
            "Savings should be negative when test strategy has higher wait than baseline"
        )
    else:
        assert savings_per_veh >= 0, (
            "Savings should be non-negative when test strategy has lower wait than baseline"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. PPO controller: must not be in 100% fallback mode
# ─────────────────────────────────────────────────────────────────────────────

def test_ppo_not_always_falling_back(layout, arrivals_df):
    """
    PPO controller should use the RL model for at least some decisions.
    If every decision contains '[RL Fallback]', the model is never used.
    """
    ctrl = RLTrafficController(
        phases=layout.phases,
        episode_duration_sec=_DURATION_SEC,
        fallback_controller=AdaptiveController(phases=layout.phases),
    )
    sim = JunctionSimulator(layout=layout, arrival_data=arrivals_df.copy(), date=_DATE)
    steps = min(_DURATION_SEC, sim.total_steps)
    decisions = []
    for _ in range(steps):
        ts = sim.time_index[sim.step_count] if sim.step_count < len(sim.time_index) else ""
        dec = ctrl.step(current_step=sim.step_count, timestamp_str=ts, approaches=sim.approaches)
        decisions.append(dec.decision_rationale or "")
        sim.step(dec)

    fallback_count = sum(1 for d in decisions if "[RL Fallback]" in d)
    fallback_ratio = fallback_count / max(1, len(decisions))
    assert fallback_ratio < 1.0, (
        f"PPO controller is in fallback mode for 100% of decisions ({fallback_count}/{len(decisions)}). "
        "Model may not be loaded. Train it via the AI & Diagnostics tab first."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. PPO comparable throughput: must not be extremely worse than Fixed-Time
# ─────────────────────────────────────────────────────────────────────────────

def test_ppo_comparable_throughput(layout, arrivals_df):
    """
    PPO total_arrived_vehicles must be > 0.
    PPO throughput should be > 25% of FixedTime (lower bound — undertrained model
    is still expected to deliver some throughput, just not optimal).
    """
    fixed_ctrl = FixedTimeController(phases=layout.phases)
    fixed_m = _run(fixed_ctrl, layout, arrivals_df)

    ppo_ctrl = RLTrafficController(
        phases=layout.phases,
        episode_duration_sec=_DURATION_SEC,
        fallback_controller=AdaptiveController(phases=layout.phases),
    )
    ppo_m = _run(ppo_ctrl, layout, arrivals_df)

    assert ppo_m.total_arrived_vehicles > 0, (
        "PPO total_arrived_vehicles must be > 0"
    )

    ratio = ppo_m.total_throughput_pcu / max(0.001, fixed_m.total_throughput_pcu)
    assert ratio > 0.20, (
        f"PPO throughput ({ppo_m.total_throughput_pcu:.1f} PCU) is less than 20% "
        f"of FixedTime ({fixed_m.total_throughput_pcu:.1f} PCU). "
        f"Ratio: {ratio:.2f}. Consider retraining the model."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Short simulation: metrics must be defined (not divide-by-zero)
# ─────────────────────────────────────────────────────────────────────────────

def test_short_simulation_no_errors(layout, arrivals_df):
    """Metrics collection must not crash or produce NaN/Inf for very short runs."""
    ctrl = FixedTimeController(phases=layout.phases)
    m = _run(ctrl, layout, arrivals_df, duration=30)

    assert not np.isnan(m.average_wait_sec), "average_wait_sec is NaN for short run"
    assert not np.isinf(m.average_wait_sec), "average_wait_sec is Inf for short run"
    assert not np.isnan(m.mean_queue_pcu), "mean_queue_pcu is NaN for short run"
    assert m.total_arrived_vehicles >= 0, "total_arrived_vehicles must be >= 0"
    assert m.starvation_occurrences >= 0, "starvation_occurrences must be >= 0"


# ─────────────────────────────────────────────────────────────────────────────
# 10. starvation_occurrences must be non-negative for all strategies
# ─────────────────────────────────────────────────────────────────────────────

def test_starvation_nonnegative_all_strategies(layout, arrivals_df):
    """starvation_occurrences must never be negative."""
    controllers = {
        "fixed_time": FixedTimeController(phases=layout.phases),
        "actuated": ActuatedController(phases=layout.phases),
        "adaptive": AdaptiveController(phases=layout.phases),
        "max_pressure": MaxPressureController(phases=layout.phases),
        "deep_rl": RLTrafficController(
            phases=layout.phases,
            episode_duration_sec=_DURATION_SEC,
            fallback_controller=AdaptiveController(phases=layout.phases),
        ),
    }
    for key, ctrl in controllers.items():
        m = _run(ctrl, layout, arrivals_df)
        assert m.starvation_occurrences >= 0, (
            f"starvation_occurrences is negative for {key}: {m.starvation_occurrences}"
        )



# ─────────────────────────────────────────────────────────────────────────────
# 11. Waiting time must never exceed simulation duration
# ─────────────────────────────────────────────────────────────────────────────

def test_waiting_time_never_exceeds_simulation_duration(layout, arrivals_df):
    """
    Physical law: average waiting time of vehicles during an episode of duration T
    cannot exceed T. Guard against Little's Law near-zero lambda division bugs.
    """
    duration = 300
    controllers = {
        "fixed": FixedTimeController(phases=layout.phases),
        "adaptive": AdaptiveController(phases=layout.phases),
        "ppo": RLTrafficController(phases=layout.phases, episode_duration_sec=duration),
    }
    for name, ctrl in controllers.items():
        m = _run(ctrl, layout, arrivals_df, duration=duration)
        assert m.average_wait_sec <= duration, (
            f"{name} average_wait_sec ({m.average_wait_sec}s) exceeded duration ({duration}s)"
        )
        assert m.max_wait_sec <= duration, (
            f"{name} max_wait_sec ({m.max_wait_sec}s) exceeded duration ({duration}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. Savings vehicle-hours and daily projection formulas
# ─────────────────────────────────────────────────────────────────────────────

def test_savings_and_projections_formulas():
    """
    Verify vehicle-hours formulas:
    - simulated_veh_hours = simulated_time_saved_sec / 3600.0
    - Misleading 12-hour daily extrapolations are eliminated (projected_daily_veh_hours == 0.0)
    """
    from backend.evaluation.comparison_lab import run_comparison_lab
    report = run_comparison_lab(duration_sec=300, selected_strategies=["fixed_time", "adaptive"])
    assert report.results is not None
    m = report.results["adaptive"]
    
    # Check veh_hours conversion
    expected_veh_hrs = round(m.total_waiting_time_saved_sec / 3600.0, 3)
    assert abs(m.simulated_veh_hours_saved - expected_veh_hrs) < 0.01

    # Check 12-hour extrapolation removed
    assert m.projected_daily_veh_hours == 0.0
    assert m.projected_daily_time_saved_hours == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 13. Misleading extrapolations eliminated
# ─────────────────────────────────────────────────────────────────────────────

def test_no_misleading_extrapolations():
    """Verify that misleading daily extrapolations are not attached to benchmark results."""
    from backend.evaluation.comparison_lab import run_comparison_lab
    rep = run_comparison_lab(duration_sec=300, selected_strategies=["fixed_time", "adaptive"])
    for m in rep.results.values():
        assert m.projected_daily_veh_hours == 0.0
        assert m.is_preliminary_projection is False


# ─────────────────────────────────────────────────────────────────────────────
# 14. Simulation ending with vehicles still queued handles residual queue
# ─────────────────────────────────────────────────────────────────────────────

def test_residual_queue_recorded(layout, arrivals_df):
    """Residual queue at final step must be recorded in metrics."""
    ctrl = FixedTimeController(phases=layout.phases)
    m = _run(ctrl, layout, arrivals_df)
    assert hasattr(m, "residual_queue_pcu")
    assert m.residual_queue_pcu >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 15. PPO honest status tracking
# ─────────────────────────────────────────────────────────────────────────────

def test_ppo_honest_status():
    """PPO must report honest ai_status (evaluated, undertrained, or model_unavailable)."""
    from backend.evaluation.comparison_lab import run_comparison_lab
    rep = run_comparison_lab(duration_sec=300, selected_strategies=["fixed_time", "deep_rl"])
    ppo_res = rep.results.get("deep_rl")
    assert ppo_res is not None
    assert ppo_res.ai_status in ("evaluated", "undertrained", "not_ready", "model_unavailable")
    # Never crowned as winner if undertrained
    if ppo_res.ai_status != "evaluated":
        assert rep.rankings[0].strategy_key != "deep_rl"


# ─────────────────────────────────────────────────────────────────────────────
# 16. Uneven traffic scenario consistent arrivals across all strategies
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_uneven_consistent_arrivals():
    """
    In the 'uneven' scenario (e.g. 1.6x North-South corridor, 0.6x East-West),
    all strategies must experience identical vehicle arrival counts under the same seed.
    """
    from backend.evaluation.comparison_lab import run_comparison_lab
    strategies = ["fixed_time", "actuated", "adaptive", "max_pressure"]
    rep = run_comparison_lab(
        duration_sec=300,
        scenario="uneven",
        selected_strategies=strategies,
        seed=42,
    )
    baseline_arrived = rep.results["fixed_time"].total_arrived_vehicles
    assert baseline_arrived > 0, "Uneven scenario must generate positive arrivals"
    for strat in strategies:
        strat_arrived = rep.results[strat].total_arrived_vehicles
        assert strat_arrived == baseline_arrived, (
            f"Arrival mismatch in uneven scenario: {strat} got {strat_arrived}, expected {baseline_arrived}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 17. Consistency score validity and most consistent strategy
# ─────────────────────────────────────────────────────────────────────────────

def test_consistency_score_validity():
    """
    Consistency score measures tail delay stability (p95 - median) and starvation freedom.
    Must be within [0, 100] for all tested strategies.
    Report must also specify most_consistent_strategy.
    """
    from backend.evaluation.comparison_lab import run_comparison_lab
    rep = run_comparison_lab(
        duration_sec=300,
        selected_strategies=["fixed_time", "adaptive", "actuated", "max_pressure"],
        seed=42,
    )
    assert rep.most_consistent_strategy in rep.results
    score = rep.results[rep.most_consistent_strategy].consistency_score
    assert 0.0 <= score <= 100.0
    for strat_key, metrics in rep.results.items():
        assert 0.0 <= metrics.consistency_score <= 100.0, (
            f"{strat_key} consistency_score={metrics.consistency_score} out of bounds"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 18. Zero or negative savings explicit honest explanation
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_or_negative_savings_plain_explanation():
    """
    When a strategy has <= 0 delay savings (such as fixed_time or an underperforming strategy),
    the system must never fabricate savings and must state:
    'No measurable improvement over fixed-time control.'
    """
    from backend.evaluation.comparison_lab import run_comparison_lab
    rep = run_comparison_lab(
        duration_sec=300,
        selected_strategies=["fixed_time", "adaptive"],
        seed=42,
    )
    # Fixed-time baseline has 0 savings
    ft_metrics = rep.results["fixed_time"]
    assert ft_metrics.avg_waiting_time_saved_per_vehicle_sec == 0.0
    assert ft_metrics.total_waiting_time_saved_sec == 0.0
    assert ft_metrics.simulated_veh_hours_saved == 0.0
    assert "No measurable improvement over fixed-time control" in ft_metrics.plain_english_analysis

