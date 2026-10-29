"""
Evaluation framework: runs both Fixed-Time and Adaptive controllers on the
same demand window and produces a fair comparison report.
"""

import datetime
import logging
from typing import Dict, List, Optional
import pandas as pd

from backend.core.schema import (
    ControllerDecision, EvaluationReport, MetricRecord, SignalState,
)
from backend.core.controller_baseline import FixedTimeController
from backend.core.controller_adaptive import AdaptiveController
from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.replay_engine import JunctionSimulator
from backend.data.loader import DatasetAdapter
from backend.core.config import FIXED_CYCLE_SPLIT_SEC

logger = logging.getLogger(__name__)

_adapter = DatasetAdapter()


def _collect_metrics(sim: JunctionSimulator) -> MetricRecord:
    if not sim.log:
        return MetricRecord(
            mean_queue_pcu=0, max_queue_pcu=0,
            average_wait_sec=0, max_wait_sec=0,
            total_throughput_vehicles=0, total_throughput_pcu=0,
            total_arrived_vehicles=0, starvation_occurrences=0,
        )

    all_queues: List[float] = []
    all_waits: List[float] = []
    starvation_count = 0
    per_approach_sum: Dict[str, Dict[str, float]] = {}

    for obs in sim.log:
        all_queues.append(obs.total_active_queue_pcu)
        all_waits.append(obs.average_delay_sec)
        for app_id, app in obs.approaches.items():
            if app.is_starving:
                starvation_count += 1
            if app_id not in per_approach_sum:
                per_approach_sum[app_id] = {
                    "total_queue": 0, "total_wait": 0, "steps": 0, "arrivals_pcu": 0,
                }
            per_approach_sum[app_id]["total_queue"] += app.estimated_queue_pcu
            per_approach_sum[app_id]["total_wait"] += app.average_wait_time_sec
            per_approach_sum[app_id]["steps"] += 1
            per_approach_sum[app_id]["arrivals_pcu"] += app.cumulative_arrivals_pcu

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
    total_arrived_veh = sum(
        a.arrivals_last_step.total_vehicles for a in final_obs.approaches.values()
    )

    return MetricRecord(
        mean_queue_pcu=round(sum(all_queues) / n, 2),
        max_queue_pcu=round(max(all_queues), 2),
        average_wait_sec=round(sum(all_waits) / n, 2),
        max_wait_sec=round(max(all_waits), 2),
        total_throughput_vehicles=int(final_obs.total_throughput_pcu / 1.0),  # rough PCU→veh
        total_throughput_pcu=round(final_obs.total_throughput_pcu, 2),
        total_arrived_vehicles=total_arrived_veh,
        starvation_occurrences=starvation_count,
        per_approach_stats=per_approach_stats,
    )


def _safe_improvement(baseline: float, adaptive: float, higher_is_better: bool = False) -> float:
    """Calculate improvement percentage safely, avoiding division by zero."""
    if baseline == 0:
        return 0.0
    if higher_is_better:
        return round(((adaptive - baseline) / baseline) * 100, 2)
    else:
        return round(((baseline - adaptive) / baseline) * 100, 2)


def run_evaluation(
    junction_id: str = "AlankarChowk",
    date: str = "11",
    duration_sec: int = 900,
    fixed_green_sec: int = FIXED_CYCLE_SPLIT_SEC,
) -> EvaluationReport:
    """
    Run both controllers on identical demand and return a comparison report.
    """
    logger.info(f"Starting evaluation: {junction_id}, date=Jan {date}, {duration_sec}s")
    layout = JunctionLayout(junction_id)

    # Load arrivals data
    cameras_needed = set(meta["camera_id"] for meta in layout.approaches.values())
    date_str = str(date)

    # Load from first available camera (primary data source)
    primary_cam = sorted(cameras_needed)[0]
    try:
        arrivals_df = _adapter.load_camera_arrivals(
            junction_id, primary_cam, date_str, max_seconds=duration_sec + 120
        )
    except FileNotFoundError:
        # Try second camera
        other_cams = sorted(cameras_needed - {primary_cam})
        if other_cams:
            arrivals_df = _adapter.load_camera_arrivals(
                junction_id, other_cams[0], date_str, max_seconds=duration_sec + 120
            )
        else:
            raise

    # --- FIXED-TIME RUN ---
    fixed_ctrl = FixedTimeController(
        phases=layout.phases,
        fixed_green_sec=fixed_green_sec,
    )
    fixed_sim = JunctionSimulator(layout=layout, arrival_data=arrivals_df, date=date_str)
    time_series_fixed = []

    for _ in range(min(duration_sec, fixed_sim.total_steps)):
        dec = fixed_ctrl.step(
            current_step=fixed_sim.step_count,
            timestamp_str=fixed_sim.time_index[fixed_sim.step_count] if fixed_sim.step_count < len(fixed_sim.time_index) else "",
            approaches=fixed_sim.approaches,
        )
        obs = fixed_sim.step(dec)
        time_series_fixed.append({
            "step": obs.step,
            "time": obs.timestamp_str,
            "queue_pcu": obs.total_active_queue_pcu,
            "throughput_pcu": obs.total_throughput_pcu,
            "delay_sec": obs.average_delay_sec,
            "phase": dec.active_phase_id,
            "signal": dec.signal_state.value,
        })

    fixed_metrics = _collect_metrics(fixed_sim)

    # --- ADAPTIVE RUN ---
    adaptive_ctrl = AdaptiveController(phases=layout.phases)
    adaptive_sim = JunctionSimulator(layout=layout, arrival_data=arrivals_df, date=date_str)
    time_series_adaptive = []

    for _ in range(min(duration_sec, adaptive_sim.total_steps)):
        dec = adaptive_ctrl.step(
            current_step=adaptive_sim.step_count,
            timestamp_str=adaptive_sim.time_index[adaptive_sim.step_count] if adaptive_sim.step_count < len(adaptive_sim.time_index) else "",
            approaches=adaptive_sim.approaches,
        )
        obs = adaptive_sim.step(dec)
        time_series_adaptive.append({
            "step": obs.step,
            "time": obs.timestamp_str,
            "queue_pcu": obs.total_active_queue_pcu,
            "throughput_pcu": obs.total_throughput_pcu,
            "delay_sec": obs.average_delay_sec,
            "phase": dec.active_phase_id,
            "signal": dec.signal_state.value,
        })

    adaptive_metrics = _collect_metrics(adaptive_sim)

    # Merge time series for report
    merged_ts = []
    for i, (f, a) in enumerate(zip(time_series_fixed, time_series_adaptive)):
        merged_ts.append({
            "step": i,
            "time": f["time"],
            "fixed_queue_pcu": f["queue_pcu"],
            "adaptive_queue_pcu": a["queue_pcu"],
            "fixed_delay_sec": f["delay_sec"],
            "adaptive_delay_sec": a["delay_sec"],
            "fixed_throughput_pcu": f["throughput_pcu"],
            "adaptive_throughput_pcu": a["throughput_pcu"],
        })

    now = datetime.datetime.now().isoformat()
    times = [r["time"] for r in time_series_fixed]

    return EvaluationReport(
        scenario_id=f"{junction_id}_jan{date}_eval",
        junction_name=junction_id,
        evaluation_duration_sec=duration_sec,
        start_time=times[0] if times else "N/A",
        end_time=times[-1] if times else "N/A",
        fixed_time_baseline=fixed_metrics,
        adaptive_controller=adaptive_metrics,
        queue_reduction_percent=_safe_improvement(
            fixed_metrics.mean_queue_pcu, adaptive_metrics.mean_queue_pcu
        ),
        waiting_time_reduction_percent=_safe_improvement(
            fixed_metrics.average_wait_sec, adaptive_metrics.average_wait_sec
        ),
        throughput_improvement_percent=_safe_improvement(
            fixed_metrics.total_throughput_pcu,
            adaptive_metrics.total_throughput_pcu,
            higher_is_better=True,
        ),
        timestamp_evaluated=now,
        summary_text=(
            f"Evaluated {junction_id} over {duration_sec}s of Jan-{date} 2023 data. "
            f"Queue reduction: {_safe_improvement(fixed_metrics.mean_queue_pcu, adaptive_metrics.mean_queue_pcu):.1f}%, "
            f"Wait reduction: {_safe_improvement(fixed_metrics.average_wait_sec, adaptive_metrics.average_wait_sec):.1f}%, "
            f"Throughput change: {_safe_improvement(fixed_metrics.total_throughput_pcu, adaptive_metrics.total_throughput_pcu, True):.1f}%."
        ),
        time_series=merged_ts,
    )

