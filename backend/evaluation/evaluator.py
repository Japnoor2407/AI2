"""
Evaluation framework: runs Fixed-Time, Actuated, and Adaptive controllers on
the same demand window and produces a fair 3-way comparison report.

Metric definitions:
  mean_queue_pcu       : time-average of total_active_queue_pcu across all steps
  max_queue_pcu        : peak total_active_queue_pcu observed
  average_wait_sec     : time-average of per-junction average_delay_sec
  max_wait_sec         : peak per-junction average_delay_sec
  median_wait_sec      : median of per-step average_delay_sec (P50)
  p95_wait_sec         : 95th-percentile of per-step average_delay_sec
  total_throughput_pcu : cumulative PCU discharged at simulation end
  residual_queue_pcu   : total_active_queue_pcu at final step (incomplete trips)
  starvation_occurrences: count of (approach × step) starvation events
  execution_time_ms    : wall-clock time for controller decisions (ms)

Improvement percentages:
  For lower-is-better metrics: ((baseline - test) / baseline) * 100
  For higher-is-better metrics: ((test - baseline) / baseline) * 100
  Zero denominator → returns None (shown as "Insufficient data" in UI)
"""

import datetime
import logging
import statistics
import time
from typing import Dict, List, Optional, Tuple

from backend.core.schema import (
    ControllerDecision, EvaluationReport, MetricRecord, SignalState,
)
from backend.core.controller_baseline import FixedTimeController
from backend.core.controller_adaptive import AdaptiveController
from backend.core.controller_actuated import ActuatedController
from backend.core.controller_rl import RLTrafficController
from backend.rl.model_manager import model_manager
from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.replay_engine import JunctionSimulator
from backend.data.loader import DatasetAdapter
from backend.core.config import FIXED_CYCLE_SPLIT_SEC

logger = logging.getLogger(__name__)

_adapter = DatasetAdapter()


def _compute_percentile(values: List[float], percentile: float) -> float:
    """Compute a percentile from a list using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (percentile / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac, 2)


def _collect_metrics(sim: JunctionSimulator, exec_time_ms: float = 0.0) -> MetricRecord:
    """
    Collect metrics from a completed simulation log.
    All values are derived from actual simulation observations.
    """
    if not sim.log:
        return MetricRecord(
            mean_queue_pcu=0, max_queue_pcu=0,
            average_wait_sec=0, max_wait_sec=0,
            median_wait_sec=0, p95_wait_sec=0,
            total_throughput_vehicles=0, total_throughput_pcu=0,
            total_arrived_vehicles=0, starvation_occurrences=0,
            residual_queue_pcu=0, controller_execution_time_ms=exec_time_ms,
        )

    all_queues: List[float] = []
    all_waits: List[float] = []
    starvation_count = 0
    per_approach_sum: Dict[str, Dict[str, float]] = {}
    total_arrived_pcu = 0.0

    for obs in sim.log:
        all_queues.append(obs.total_active_queue_pcu)
        all_waits.append(obs.average_delay_sec)
        for app_id, app in obs.approaches.items():
            if app.is_starving:
                starvation_count += 1
            if app_id not in per_approach_sum:
                per_approach_sum[app_id] = {
                    "total_queue": 0.0, "total_wait": 0.0, "steps": 0, "arrivals_pcu": 0.0,
                }
            per_approach_sum[app_id]["total_queue"] += app.estimated_queue_pcu
            per_approach_sum[app_id]["total_wait"] += app.average_wait_time_sec
            per_approach_sum[app_id]["steps"] += 1
            per_approach_sum[app_id]["arrivals_pcu"] += app.arrivals_last_step.to_pcu()
            total_arrived_pcu += app.arrivals_last_step.to_pcu()

    n = max(1, len(sim.log))
    per_approach_stats = {
        app_id: {
            "mean_queue_pcu": round(v["total_queue"] / max(1, v["steps"]), 2),
            "mean_wait_sec": round(v["total_wait"] / max(1, v["steps"]), 2),
            "total_arrivals_pcu": round(v["arrivals_pcu"], 2),
        }
        for app_id, v in per_approach_sum.items()
    }

    final_obs = sim.log[-1]
    residual_queue = final_obs.total_active_queue_pcu

    # Fix: total_arrived_vehicles uses cumulative_arrivals_pcu from approach state,
    # NOT arrivals_last_step (which only measures the single final second).
    # Convert PCU to vehicles using Indian mixed-traffic average ~0.8 PCU/veh.
    cumulative_arr_pcu = sum(
        a.cumulative_arrivals_pcu for a in final_obs.approaches.values()
    )
    total_arrived_veh = max(0, int(round(cumulative_arr_pcu / 0.8)))

    # Vehicles cleared: cumulative departures converted to vehicles
    cumulative_dep_pcu = sum(
        a.cumulative_departures_pcu for a in final_obs.approaches.values()
    )
    total_processed_veh = max(0, int(round(cumulative_dep_pcu / 0.8)))

    # Fix: starvation counted as EVENTS (approach first crosses threshold),
    # not per-step flag (which multiplies counts by 100s during holdover).
    starvation_events = 0
    prev_starving: dict = {}
    for obs in sim.log:
        for app_id, app in obs.approaches.items():
            was_starving = prev_starving.get(app_id, False)
            if app.is_starving and not was_starving:
                starvation_events += 1
            prev_starving[app_id] = app.is_starving

    return MetricRecord(
        mean_queue_pcu=round(sum(all_queues) / n, 2),
        max_queue_pcu=round(max(all_queues), 2),
        average_wait_sec=round(sum(all_waits) / n, 2),
        max_wait_sec=round(max(all_waits), 2),
        median_wait_sec=_compute_percentile(all_waits, 50),
        p95_wait_sec=_compute_percentile(all_waits, 95),
        total_throughput_vehicles=total_processed_veh,
        total_throughput_pcu=round(final_obs.total_throughput_pcu, 2),
        total_arrived_vehicles=total_arrived_veh,
        starvation_occurrences=starvation_events,
        residual_queue_pcu=round(residual_queue, 2),
        controller_execution_time_ms=round(exec_time_ms, 1),
        per_approach_stats=per_approach_stats,
    )


def _safe_improvement(
    baseline: float, test: float, higher_is_better: bool = False
) -> Optional[float]:
    """
    Calculate improvement percentage safely.
    Returns None if baseline is 0 (undefined).
    Positive = test is better than baseline.
    """
    if baseline == 0:
        return None
    if higher_is_better:
        return round(((test - baseline) / baseline) * 100, 2)
    else:
        return round(((baseline - test) / baseline) * 100, 2)


def _run_controller(
    ctrl,
    layout: JunctionLayout,
    arrivals_df,
    date_str: str,
    duration_sec: int,
) -> Tuple[JunctionSimulator, float]:
    """Run a single controller for duration_sec steps and return sim + exec time ms."""
    sim = JunctionSimulator(layout=layout, arrival_data=arrivals_df, date=date_str)
    total_ctrl_ms = 0.0

    steps = min(duration_sec, sim.total_steps)
    for _ in range(steps):
        ts = sim.time_index[sim.step_count] if sim.step_count < len(sim.time_index) else ""
        t0 = time.perf_counter()
        dec = ctrl.step(
            current_step=sim.step_count,
            timestamp_str=ts,
            approaches=sim.approaches,
        )
        total_ctrl_ms += (time.perf_counter() - t0) * 1000.0
        sim.step(dec)

    return sim, total_ctrl_ms


def _build_time_series(sim: JunctionSimulator) -> List[dict]:
    return [
        {
            "step": obs.step,
            "time": obs.timestamp_str,
            "queue_pcu": obs.total_active_queue_pcu,
            "throughput_pcu": obs.total_throughput_pcu,
            "delay_sec": obs.average_delay_sec,
            "phase": obs.controller_decision.active_phase_id,
            "signal": obs.controller_decision.signal_state.value,
        }
        for obs in sim.log
    ]


def run_evaluation(
    junction_id: str = "AlankarChowk",
    date: str = "11",
    duration_sec: int = 900,
    fixed_green_sec: int = FIXED_CYCLE_SPLIT_SEC,
    scenario: str = "normal",
    demand_multiplier: float = 1.0,
    seed: Optional[int] = None,
    include_rl: Optional[bool] = None,
) -> EvaluationReport:
    """
    Run Fixed-Time, Actuated, and Adaptive controllers on identical demand
    and return a 3-way comparison report.
    """
    logger.info(
        f"Starting 3-controller evaluation: {junction_id}, Jan-{date}, "
        f"{duration_sec}s, scenario={scenario}, demand_mult={demand_multiplier}"
    )
    layout = JunctionLayout(junction_id)

    # Load arrivals data from primary camera
    cameras_needed = set(meta["camera_id"] for meta in layout.approaches.values())
    primary_cam = sorted(cameras_needed)[0]
    try:
        arrivals_df = _adapter.load_camera_arrivals(
            junction_id, primary_cam, str(date), max_seconds=duration_sec + 120
        )
    except FileNotFoundError:
        other_cams = sorted(cameras_needed - {primary_cam})
        if other_cams:
            arrivals_df = _adapter.load_camera_arrivals(
                junction_id, other_cams[0], str(date), max_seconds=duration_sec + 120
            )
        else:
            raise

    # Apply demand multiplier if scenario requires scaling
    if demand_multiplier != 1.0:
        arrivals_df = _scale_arrivals_df(arrivals_df, demand_multiplier)

    date_str = str(date)

    # --- FIXED-TIME RUN ---
    fixed_ctrl = FixedTimeController(phases=layout.phases, fixed_green_sec=fixed_green_sec)
    fixed_sim, fixed_exec_ms = _run_controller(fixed_ctrl, layout, arrivals_df, date_str, duration_sec)
    fixed_metrics = _collect_metrics(fixed_sim, fixed_exec_ms)
    time_series_fixed = _build_time_series(fixed_sim)

    # --- ACTUATED RUN ---
    actuated_ctrl = ActuatedController(phases=layout.phases)
    actuated_sim, actuated_exec_ms = _run_controller(actuated_ctrl, layout, arrivals_df, date_str, duration_sec)
    actuated_metrics = _collect_metrics(actuated_sim, actuated_exec_ms)
    time_series_actuated = _build_time_series(actuated_sim)

    # --- ADAPTIVE RUN ---
    adaptive_ctrl = AdaptiveController(phases=layout.phases)
    adaptive_sim, adaptive_exec_ms = _run_controller(adaptive_ctrl, layout, arrivals_df, date_str, duration_sec)
    adaptive_metrics = _collect_metrics(adaptive_sim, adaptive_exec_ms)
    time_series_adaptive = _build_time_series(adaptive_sim)

    # --- PPO RL RUN (Optional / Auto-detected) ---
    should_run_rl = include_rl if include_rl is not None else model_manager.has_trained_model()
    rl_metrics = None
    time_series_rl = []
    if should_run_rl:
        try:
            rl_ctrl = RLTrafficController(phases=layout.phases, episode_duration_sec=duration_sec)
            rl_sim, rl_exec_ms = _run_controller(rl_ctrl, layout, arrivals_df, date_str, duration_sec)
            rl_metrics = _collect_metrics(rl_sim, rl_exec_ms)
            time_series_rl = _build_time_series(rl_sim)
            logger.info("Evaluation: PPO RL run completed successfully.")
        except Exception as e:
            logger.warning(f"Evaluation: PPO RL run failed ({e}). Proceeding without RL metrics.")

    # Merge time series for comparative chart
    merged_ts = []
    for i, (f, u, a) in enumerate(
        zip(time_series_fixed, time_series_actuated, time_series_adaptive)
    ):
        pt = {
            "step": i,
            "time": f["time"],
            "fixed_queue_pcu": f["queue_pcu"],
            "actuated_queue_pcu": u["queue_pcu"],
            "adaptive_queue_pcu": a["queue_pcu"],
            "fixed_delay_sec": f["delay_sec"],
            "actuated_delay_sec": u["delay_sec"],
            "adaptive_delay_sec": a["delay_sec"],
            "fixed_throughput_pcu": f["throughput_pcu"],
            "actuated_throughput_pcu": u["throughput_pcu"],
            "adaptive_throughput_pcu": a["throughput_pcu"],
        }
        if i < len(time_series_rl):
            r = time_series_rl[i]
            pt["rl_queue_pcu"] = r["queue_pcu"]
            pt["rl_delay_sec"] = r["delay_sec"]
            pt["rl_throughput_pcu"] = r["throughput_pcu"]
        merged_ts.append(pt)

    now = datetime.datetime.now().isoformat()
    times = [r["time"] for r in time_series_fixed]

    # Improvements vs Fixed-Time baseline
    imp_queue_vs_fixed = _safe_improvement(fixed_metrics.mean_queue_pcu, adaptive_metrics.mean_queue_pcu)
    imp_wait_vs_fixed = _safe_improvement(fixed_metrics.average_wait_sec, adaptive_metrics.average_wait_sec)
    imp_thru_vs_fixed = _safe_improvement(
        fixed_metrics.total_throughput_pcu, adaptive_metrics.total_throughput_pcu, higher_is_better=True
    )

    # Improvements vs Actuated baseline
    imp_queue_vs_actuated = _safe_improvement(actuated_metrics.mean_queue_pcu, adaptive_metrics.mean_queue_pcu)
    imp_wait_vs_actuated = _safe_improvement(actuated_metrics.average_wait_sec, adaptive_metrics.average_wait_sec)
    imp_thru_vs_actuated = _safe_improvement(
        actuated_metrics.total_throughput_pcu, adaptive_metrics.total_throughput_pcu, higher_is_better=True
    )

    summary = (
        f"3-controller evaluation: {junction_id} · Jan-{date} 2023 · {duration_sec}s · scenario={scenario}. "
        f"Adaptive vs Fixed: queue {_fmt_pct(imp_queue_vs_fixed)}, "
        f"wait {_fmt_pct(imp_wait_vs_fixed)}, "
        f"throughput {_fmt_pct(imp_thru_vs_fixed)}. "
        f"Adaptive vs Actuated: queue {_fmt_pct(imp_queue_vs_actuated)}, "
        f"wait {_fmt_pct(imp_wait_vs_actuated)}."
    )

    # RL Improvements vs Fixed-Time baseline
    imp_rl_vs_fixed_queue = _safe_improvement(fixed_metrics.mean_queue_pcu, rl_metrics.mean_queue_pcu) if rl_metrics else None
    imp_rl_vs_fixed_wait = _safe_improvement(fixed_metrics.average_wait_sec, rl_metrics.average_wait_sec) if rl_metrics else None
    imp_rl_vs_fixed_thru = _safe_improvement(
        fixed_metrics.total_throughput_pcu, rl_metrics.total_throughput_pcu, higher_is_better=True
    ) if rl_metrics else None

    return EvaluationReport(
        scenario_id=f"{junction_id}_jan{date}_{scenario}",
        scenario_name=scenario,
        junction_name=junction_id,
        evaluation_duration_sec=duration_sec,
        start_time=times[0] if times else "N/A",
        end_time=times[-1] if times else "N/A",
        random_seed=seed,
        fixed_time_baseline=fixed_metrics,
        actuated_baseline=actuated_metrics,
        adaptive_controller=adaptive_metrics,
        deep_rl_controller=rl_metrics,
        queue_reduction_percent=imp_queue_vs_fixed or 0.0,
        waiting_time_reduction_percent=imp_wait_vs_fixed or 0.0,
        throughput_improvement_percent=imp_thru_vs_fixed or 0.0,
        improvements_vs_actuated={
            "queue_reduction_percent": imp_queue_vs_actuated,
            "waiting_time_reduction_percent": imp_wait_vs_actuated,
            "throughput_improvement_percent": imp_thru_vs_actuated,
        },
        improvements_vs_rl={
            "rl_vs_fixed_queue_reduction_percent": imp_rl_vs_fixed_queue,
            "rl_vs_fixed_wait_reduction_percent": imp_rl_vs_fixed_wait,
            "rl_vs_fixed_throughput_improvement_percent": imp_rl_vs_fixed_thru,
        } if rl_metrics else None,
        timestamp_evaluated=now,
        summary_text=summary,
        time_series=merged_ts,
    )


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _scale_arrivals_df(df, multiplier: float):
    """Scale all vehicle count columns by demand multiplier."""
    import pandas as pd
    df = df.copy()
    count_cols = [c for c in df.columns if c in (
        "car", "motorbike", "bus", "truck",
        "car_arrivals", "motorbike_arrivals", "bus_arrivals", "truck_arrivals",
        "pcu_arrivals",
    )]
    for col in count_cols:
        df[col] = (df[col] * multiplier).round().astype(int)
    return df
