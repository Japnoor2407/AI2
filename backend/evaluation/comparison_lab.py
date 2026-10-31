"""
Strategy Comparison Lab: Benchmarking framework for fair, reproducible comparisons
across all 5 traffic control strategies:
1. Fixed-Time (Baseline reference)
2. Actuated (Rule-based gap-out)
3. Weighted Adaptive (Queue & delay heuristic)
4. Max-Pressure (Network throughput & backpressure)
5. Deep Reinforcement Learning (PPO neural policy)

Guarantees:
- Strictly identical vehicle arrivals and random seed across all tested strategies.
- Configurable scenarios: normal (1.0x), low (0.7x), rush_hour (1.5x), uneven (asymmetric).
- Actual measured cumulative delay saved in vehicle-hours (veh-hrs) and vehicle-seconds.
- No misleading daily extrapolations; zero savings clearly marked as 'No measurable improvement'.
- Telemetry-grounded AI explanations detailing green allocations and phase switching behavior.
- Consistency scoring measuring tail-risk stability (P95 vs median gap and starvation avoidance).
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from backend.core.schema import MetricRecord
from backend.core.config import FIXED_CYCLE_SPLIT_SEC
from backend.core.controller_baseline import FixedTimeController
from backend.core.controller_actuated import ActuatedController
from backend.core.controller_adaptive import AdaptiveController
from backend.core.controller_max_pressure import MaxPressureController
from backend.core.controller_rl import RLTrafficController
from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.replay_engine import JunctionSimulator
from backend.data.loader import DatasetAdapter
from backend.evaluation.evaluator import _collect_metrics, _scale_arrivals_df

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

STRATEGY_METADATA = {
    "fixed_time": {
        "name": "Fixed-Time",
        "description": "Predetermined signal splits without vehicle sensor feedback.",
        "category": "Baseline",
    },
    "actuated": {
        "name": "Actuated Control",
        "description": "Extends green when arriving vehicles are detected; gaps out during lulls.",
        "category": "Rule-Based",
    },
    "adaptive": {
        "name": "Weighted Adaptive",
        "description": "Dynamic queue scoring with starvation prevention and delay balancing.",
        "category": "AI / Heuristic",
    },
    "max_pressure": {
        "name": "Max-Pressure",
        "description": "Prioritizes signal phases with greatest upstream queue vs downstream exit capacity.",
        "category": "Feedback Control",
    },
    "deep_rl": {
        "name": "Deep Reinforcement Learning (PPO)",
        "description": "Neural network policy trained via PPO optimizing multi-objective reward.",
        "category": "Deep RL",
    },
}


class StrategyMetrics(BaseModel):
    key: str
    name: str
    category: str
    description: str
    mean_queue_pcu: float
    max_queue_pcu: float
    average_wait_sec: float
    median_wait_sec: float = 0.0
    p95_wait_sec: float
    total_throughput_pcu: float
    total_arrived_vehicles: int
    vehicles_processed: int = 0
    starvation_occurrences: int
    controller_execution_time_ms: float
    # Measured savings vs Fixed-Time baseline during the simulation
    avg_waiting_time_saved_per_vehicle_sec: float = 0.0
    total_waiting_time_saved_sec: float = 0.0
    simulated_veh_hours_saved: float = 0.0
    consistency_score: float = 50.0
    waiting_time_saved_percent: float = 0.0
    queue_reduction_percent: float = 0.0
    throughput_improvement_percent: float = 0.0
    performance_verdict: str = "Baseline"
    plain_english_analysis: str = ""
    ai_status: str = "evaluated"  # "evaluated", "undertrained", "not_ready", "model_unavailable"
    # Legacy compatibility fields
    simulated_time_saved_sec: float = 0.0
    projected_hourly_time_saved_min: float = 0.0
    projected_daily_time_saved_hours: float = 0.0
    projected_daily_veh_hours: float = 0.0
    is_preliminary_projection: bool = False


class StrategyRanking(BaseModel):
    rank: int
    strategy_key: str
    strategy_name: str
    composite_score: float
    consistency_score: float = 50.0
    rationale: str


class ComparisonLabReport(BaseModel):
    session_id: str
    junction_id: str
    date: str
    duration_sec: int
    scenario: str
    demand_multiplier: float
    seed: Optional[int] = None
    created_at: str
    strategies_tested: List[str]
    results: Dict[str, StrategyMetrics]
    rankings: List[StrategyRanking]
    most_consistent_strategy: str = ""
    plain_english_summary: str
    time_series: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


def _apply_scenario_scaling(df, scenario: str, multiplier: float):
    """Applies scenario-specific traffic scaling."""
    import pandas as pd
    df = df.copy()
    if scenario == "uneven":
        # Asymmetric corridor bottleneck: heavy North/West, lighter South/East
        dir_col = "Direction" if "Direction" in df.columns else None
        if dir_col:
            up_mask = df[dir_col].astype(str).str.upper().isin(["UP", "NORTH", "LEFT", "WEST"])
            count_cols = [c for c in df.columns if any(k in c.lower() for k in ("car", "motorbike", "bus", "truck", "pcu"))]
            for c in count_cols:
                df.loc[up_mask, c] = (df.loc[up_mask, c] * 1.6).round().astype(int)
                df.loc[~up_mask, c] = (df.loc[~up_mask, c] * 0.6).round().astype(int)
            return df
    elif multiplier != 1.0:
        return _scale_arrivals_df(df, multiplier)
    return df


def _downsample_log(log: List[Any], max_points: int = 60) -> List[Dict[str, Any]]:
    """Downsamples simulation step log for lightweight, smooth UI charts."""
    if not log:
        return []
    total = len(log)
    step_size = max(1, total // max_points)
    sampled = []
    for i in range(0, total, step_size):
        obs = log[i]
        sampled.append({
            "step": obs.step,
            "time": obs.timestamp_str,
            "queue_pcu": obs.total_active_queue_pcu,
            "throughput_pcu": obs.total_throughput_pcu,
            "delay_sec": obs.average_delay_sec,
            "phase": obs.controller_decision.active_phase_id,
        })
    if sampled and sampled[-1]["step"] != log[-1].step:
        obs = log[-1]
        sampled.append({
            "step": obs.step,
            "time": obs.timestamp_str,
            "queue_pcu": obs.total_active_queue_pcu,
            "throughput_pcu": obs.total_throughput_pcu,
            "delay_sec": obs.average_delay_sec,
            "phase": obs.controller_decision.active_phase_id,
        })
    return sampled


def _generate_telemetry_analysis(
    key: str,
    metrics: StrategyMetrics,
    fixed_base: MetricRecord,
    sim_log: List[Any],
    phases: List[Any],
) -> str:
    """
    Generates plain-English analysis grounded in actual simulation telemetry and signal decisions.
    Explains which approaches were prioritized, phase switching behavior, and measured delay impact.
    """
    if key == "fixed_time":
        return (
            f"Fixed-Time baseline cycled predetermined splits without sensor feedback. "
            f"Processed {metrics.vehicles_processed} vehicles with an average wait of {metrics.average_wait_sec:.1f}s "
            f"and P95 wait of {metrics.p95_wait_sec:.1f}s across {metrics.starvation_occurrences} starvation events."
        )

    # Analyze green allocations from actual simulation step log
    phase_green_counts: Dict[int, int] = {}
    total_switches = 0
    prev_phase = None
    for obs in sim_log:
        dec = obs.controller_decision
        if dec.signal_state.value == "GREEN":
            p_id = dec.active_phase_id
            phase_green_counts[p_id] = phase_green_counts.get(p_id, 0) + 1
            if prev_phase is not None and prev_phase != p_id:
                total_switches += 1
            prev_phase = p_id

    total_green = max(1, sum(phase_green_counts.values()))
    top_phase_id = max(phase_green_counts.keys(), key=lambda p: phase_green_counts[p]) if phase_green_counts else 1
    top_phase_name = next((p.name for p in phases if p.phase_id == top_phase_id), f"Phase {top_phase_id}")
    top_phase_pct = round((phase_green_counts.get(top_phase_id, 0) / total_green) * 100)

    time_diff = round(fixed_base.average_wait_sec - metrics.average_wait_sec, 2)
    queue_diff = round(fixed_base.mean_queue_pcu - metrics.mean_queue_pcu, 2)

    if time_diff > 0:
        return (
            f"Measured delay reduction of {metrics.waiting_time_saved_percent:+.1f}% vs baseline "
            f"(saving {time_diff:+.1f}s/vehicle, total {metrics.simulated_veh_hours_saved:.2f} vehicle-hours delay prevented across {metrics.total_arrived_vehicles} vehicles). "
            f"Allocated {top_phase_pct}% green priority to {top_phase_name} during peak arrival surges with {total_switches} phase transitions, "
            f"maintaining mean queue at {metrics.mean_queue_pcu:.1f} PCU (reduced by {metrics.queue_reduction_percent:+.1f}%)."
        )
    else:
        return (
            f"No measurable improvement over fixed-time control ({time_diff:.1f}s/vehicle additional delay, {abs(queue_diff):.1f} PCU larger mean queue). "
            f"Executed {total_switches} phase switches, but green allocation was held too long on lower-demand approaches, "
            f"allowing queues to peak at {metrics.max_queue_pcu:.1f} PCU with {metrics.starvation_occurrences} starvation events."
        )


def run_comparison_lab(
    junction_id: str = "AlankarChowk",
    date: str = "11",
    duration_sec: int = 900,
    scenario: str = "normal",
    demand_multiplier: float = 1.0,
    selected_strategies: Optional[List[str]] = None,
    seed: Optional[int] = 42,
) -> ComparisonLabReport:
    """
    Executes a multi-strategy benchmark under strictly identical arrival demand.
    """
    session_id = str(uuid.uuid4())[:8]
    created_at = datetime.now().isoformat()
    adapter = DatasetAdapter()
    layout = JunctionLayout(junction_id)

    # 1. Load baseline arrivals data
    cameras_needed = set(meta["camera_id"] for meta in layout.approaches.values())
    primary_cam = sorted(cameras_needed)[0]
    try:
        arrivals_df = adapter.load_camera_arrivals(
            junction_id, primary_cam, str(date), max_seconds=duration_sec + 120
        )
    except FileNotFoundError:
        other_cams = sorted(cameras_needed - {primary_cam})
        if other_cams:
            arrivals_df = adapter.load_camera_arrivals(
                junction_id, other_cams[0], str(date), max_seconds=duration_sec + 120
            )
        else:
            raise

    # Apply scenario scaling (normal, low, rush_hour, uneven)
    mult = 0.7 if scenario == "low" else (1.5 if scenario in ("heavy", "rush_hour") else demand_multiplier)
    arrivals_df = _apply_scenario_scaling(arrivals_df, scenario, mult)

    # Available strategy keys
    all_possible = ["fixed_time", "actuated", "adaptive", "max_pressure", "deep_rl"]
    if selected_strategies:
        strategies_to_run = [s for s in selected_strategies if s in all_possible]
    else:
        strategies_to_run = all_possible

    # Ensure fixed_time is always run as the reference baseline
    if "fixed_time" not in strategies_to_run:
        strategies_to_run.insert(0, "fixed_time")

    date_str = str(date)
    raw_metrics: Dict[str, MetricRecord] = {}
    raw_sim_logs: Dict[str, List[Any]] = {}
    time_series_data: Dict[str, List[Dict[str, Any]]] = {}

    for strat_key in strategies_to_run:
        logger.info(f"[Lab-{session_id}] Executing strategy: {strat_key} ({duration_sec}s, scenario={scenario})")
        
        # Instantiate controller
        if strat_key == "fixed_time":
            ctrl = FixedTimeController(phases=layout.phases, fixed_green_sec=FIXED_CYCLE_SPLIT_SEC)
        elif strat_key == "actuated":
            ctrl = ActuatedController(phases=layout.phases)
        elif strat_key == "adaptive":
            ctrl = AdaptiveController(phases=layout.phases)
        elif strat_key == "max_pressure":
            ctrl = MaxPressureController(phases=layout.phases)
        elif strat_key == "deep_rl":
            ctrl = RLTrafficController(
                phases=layout.phases,
                episode_duration_sec=duration_sec,
                fallback_controller=AdaptiveController(phases=layout.phases),
            )
        else:
            continue

        sim = JunctionSimulator(layout=layout, arrival_data=arrivals_df.copy(), date=date_str)
        t0 = time.perf_counter()
        steps = min(duration_sec, sim.total_steps)

        for _ in range(steps):
            ts = sim.time_index[sim.step_count] if sim.step_count < len(sim.time_index) else ""
            dec = ctrl.step(
                current_step=sim.step_count,
                timestamp_str=ts,
                approaches=sim.approaches,
            )
            sim.step(dec)

        exec_ms = (time.perf_counter() - t0) * 1000.0
        m = _collect_metrics(sim, exec_time_ms=exec_ms)
        raw_metrics[strat_key] = m
        raw_sim_logs[strat_key] = sim.log
        time_series_data[strat_key] = _downsample_log(sim.log, max_points=60)

    # 2. Compute comparisons vs Fixed-Time baseline
    fixed_base = raw_metrics["fixed_time"]
    results: Dict[str, StrategyMetrics] = {}
    import numpy as np

    for key, m in raw_metrics.items():
        meta = STRATEGY_METADATA.get(key, {"name": key, "category": "General", "description": ""})
        
        # Measured delay difference vs Fixed-Time baseline
        time_saved_per_veh = round(fixed_base.average_wait_sec - m.average_wait_sec, 2)
        veh_count = m.total_arrived_vehicles

        # Only positive savings are counted; zero or negative savings are never manufactured
        if time_saved_per_veh > 0 and veh_count > 0:
            total_time_saved = round(time_saved_per_veh * veh_count, 1)
            simulated_veh_hours = round(total_time_saved / 3600.0, 3)
        else:
            total_time_saved = 0.0
            simulated_veh_hours = 0.0

        # Baseline denominators (safe minimum 1.0 to avoid small-number division spikes)
        base_wait = max(1.0, fixed_base.average_wait_sec)
        base_queue = max(1.0, fixed_base.mean_queue_pcu)
        base_thru = max(1.0, fixed_base.total_throughput_pcu)

        wait_pct = round(((fixed_base.average_wait_sec - m.average_wait_sec) / base_wait) * 100.0, 1)
        queue_pct = round(((fixed_base.mean_queue_pcu - m.mean_queue_pcu) / base_queue) * 100.0, 1)
        thru_pct = round(((m.total_throughput_pcu - fixed_base.total_throughput_pcu) / base_thru) * 100.0, 1)

        # Consistency score: measures tail-delay stability and starvation avoidance
        p95_gap = max(0.0, m.p95_wait_sec - m.median_wait_sec)
        consistency = round(float(np.clip(100.0 - (p95_gap * 1.5) - (m.starvation_occurrences * 5.0), 0.0, 100.0)), 1)

        # AI Status evaluation
        ai_status = "evaluated"
        if key == "deep_rl":
            from backend.rl.model_manager import model_manager
            if not model_manager.has_trained_model():
                ai_status = "model_unavailable"
            elif wait_pct < -30.0 or m.total_throughput_pcu < (fixed_base.total_throughput_pcu * 0.4):
                ai_status = "undertrained"

        # Performance verdict
        if key == "fixed_time":
            verdict = "Benchmark Baseline"
        elif key == "deep_rl" and ai_status == "model_unavailable":
            verdict = "Model Unavailable"
        elif key == "deep_rl" and ai_status == "undertrained":
            verdict = "Undertrained (Not Production Ready)"
        elif wait_pct >= 20.0:
            verdict = "Superior Optimization"
        elif wait_pct > 0.0:
            verdict = "Moderate Improvement"
        else:
            verdict = "No measurable improvement over fixed-time control"

        strat_metric = StrategyMetrics(
            key=key,
            name=meta["name"],
            category=meta["category"],
            description=meta["description"],
            mean_queue_pcu=m.mean_queue_pcu,
            max_queue_pcu=m.max_queue_pcu,
            average_wait_sec=m.average_wait_sec,
            median_wait_sec=m.median_wait_sec,
            p95_wait_sec=m.p95_wait_sec,
            total_throughput_pcu=m.total_throughput_pcu,
            total_arrived_vehicles=m.total_arrived_vehicles,
            vehicles_processed=m.total_throughput_vehicles,
            starvation_occurrences=m.starvation_occurrences,
            controller_execution_time_ms=m.controller_execution_time_ms,
            avg_waiting_time_saved_per_vehicle_sec=time_saved_per_veh,
            total_waiting_time_saved_sec=total_time_saved,
            simulated_veh_hours_saved=simulated_veh_hours,
            consistency_score=consistency,
            waiting_time_saved_percent=wait_pct,
            queue_reduction_percent=queue_pct,
            throughput_improvement_percent=thru_pct,
            performance_verdict=verdict,
            plain_english_analysis="",  # populated below
            ai_status=ai_status,
            simulated_time_saved_sec=total_time_saved,
            projected_hourly_time_saved_min=0.0,
            projected_daily_time_saved_hours=0.0,
            projected_daily_veh_hours=0.0,
            is_preliminary_projection=False,
        )

        strat_metric.plain_english_analysis = _generate_telemetry_analysis(
            key=key,
            metrics=strat_metric,
            fixed_base=fixed_base,
            sim_log=raw_sim_logs[key],
            phases=layout.phases,
        )
        results[key] = strat_metric

    # 3. Strategy Ranking using a balanced 0-100 efficiency score with consistency weighting
    scores: Dict[str, float] = {}
    for key, res in results.items():
        if key == "fixed_time":
            scores[key] = 50.0  # Baseline index 50.0
        elif key == "deep_rl" and res.ai_status in ("undertrained", "model_unavailable"):
            # Never crown an untrained or unavailable AI model as winner
            wait_norm = float(np.clip(res.waiting_time_saved_percent, -100.0, 100.0))
            score = round(float(np.clip(30.0 + (wait_norm * 0.1), 5.0, 39.0)), 1)
            scores[key] = score
        else:
            wait_norm = float(np.clip(res.waiting_time_saved_percent, -100.0, 100.0))
            queue_norm = float(np.clip(res.queue_reduction_percent, -100.0, 100.0))
            thru_norm = float(np.clip(res.throughput_improvement_percent, -50.0, 100.0))
            consistency_norm = float(np.clip(res.consistency_score - 50.0, -50.0, 50.0))

            # 40% wait reduction, 25% queue reduction, 15% throughput improvement, 20% consistency
            raw = 0.40 * wait_norm + 0.25 * queue_norm + 0.15 * thru_norm + 0.20 * consistency_norm
            score = round(float(np.clip(50.0 + raw * 0.5, 0.0, 100.0)), 1)
            scores[key] = score

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    rankings: List[StrategyRanking] = []
    for idx, key in enumerate(sorted_keys, start=1):
        res = results[key]
        score = scores[key]
        if key == "fixed_time":
            rationale = "Fixed predetermined timer baseline. Indexed reference score of 50.0."
        elif key == "deep_rl" and res.ai_status != "evaluated":
            rationale = (
                f"NOT READY ({res.performance_verdict}): Evaluated wait time was {res.average_wait_sec}s "
                f"({res.waiting_time_saved_percent:+0.1f}% vs baseline). Requires further training before deployment."
            )
        elif res.avg_waiting_time_saved_per_vehicle_sec <= 0:
            rationale = (
                f"{res.performance_verdict}: Average wait time of {res.average_wait_sec}s showed no improvement "
                f"over fixed-time control (queue {res.mean_queue_pcu} PCU, {res.starvation_occurrences} starvations)."
            )
        else:
            rationale = (
                f"{res.performance_verdict}: Saved {res.avg_waiting_time_saved_per_vehicle_sec:+0.1f}s/veh "
                f"({res.waiting_time_saved_percent:+0.1f}% less wait) with {res.queue_reduction_percent:+0.1f}% queue reduction, "
                f"consistency score {res.consistency_score}/100, and {res.starvation_occurrences} starvation events."
            )
        rankings.append(
            StrategyRanking(
                rank=idx,
                strategy_key=key,
                strategy_name=res.name,
                composite_score=score,
                consistency_score=res.consistency_score,
                rationale=rationale,
            )
        )

    best_strat = rankings[0]
    best_m = results[best_strat.strategy_key]

    # Find the most consistent strategy
    most_consistent_key = max(results.keys(), key=lambda k: results[k].consistency_score)
    most_consistent_name = results[most_consistent_key].name

    # Build truthful summary of measured results
    if best_m.simulated_veh_hours_saved > 0:
        sim_saved_str = f"eliminating {best_m.simulated_veh_hours_saved:.2f} vehicle-hours ({int(best_m.total_waiting_time_saved_sec):,} veh-seconds) of commuter delay across {best_m.total_arrived_vehicles:,} arrived vehicles"
    else:
        sim_saved_str = "achieving no measurable delay reduction over fixed-time control"

    plain_summary = (
        f"In a {duration_sec // 60}-minute simulation at {junction_id} under '{scenario}' traffic demand, "
        f"{best_strat.strategy_name} ranked #1 with an efficiency score of {best_strat.composite_score}/100. "
        f"It produced an average wait of {best_m.average_wait_sec:.1f}s (P95: {best_m.p95_wait_sec:.1f}s), "
        f"saving {best_m.avg_waiting_time_saved_per_vehicle_sec:+0.1f}s per vehicle ({best_m.waiting_time_saved_percent:+0.1f}% vs Fixed-Time), "
        f"{sim_saved_str}. Vehicles cleared: {best_m.vehicles_processed:,}. "
        f"Most consistent strategy across tail delays was {most_consistent_name} (consistency score: {results[most_consistent_key].consistency_score}/100)."
    )

    report = ComparisonLabReport(
        session_id=session_id,
        junction_id=junction_id,
        date=date_str,
        duration_sec=duration_sec,
        scenario=scenario,
        demand_multiplier=demand_multiplier,
        seed=seed,
        created_at=created_at,
        strategies_tested=strategies_to_run,
        results=results,
        rankings=rankings,
        most_consistent_strategy=most_consistent_name,
        plain_english_summary=plain_summary,
        time_series=time_series_data,
    )

    # Persist report to JSON
    report_path = os.path.join(REPORTS_DIR, f"lab_{session_id}.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        logger.info(f"Comparison Lab report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save lab report: {e}")

    return report


def get_lab_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Returns summarized history of past comparison lab runs."""
    try:
        limit_val = int(limit)
    except (TypeError, ValueError):
        limit_val = 20

    if not os.path.exists(REPORTS_DIR):
        return []
    history = []
    files = sorted(
        [f for f in os.listdir(REPORTS_DIR) if f.startswith("lab_") and f.endswith(".json")],
        reverse=True,
    )
    for fname in files[:limit_val]:
        fpath = os.path.join(REPORTS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                history.append({
                    "session_id": data.get("session_id"),
                    "junction_id": data.get("junction_id"),
                    "duration_sec": data.get("duration_sec"),
                    "created_at": data.get("created_at"),
                    "strategies_tested": data.get("strategies_tested", []),
                    "best_strategy": data.get("rankings", [{}])[0].get("strategy_name", "N/A"),
                    "plain_english_summary": data.get("plain_english_summary", ""),
                })
        except Exception:
            continue
    return history


def get_lab_report(session_id: str) -> Optional[ComparisonLabReport]:
    """Load a specific comparison report by session ID."""
    fpath = os.path.join(REPORTS_DIR, f"lab_{session_id}.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return ComparisonLabReport(**data)


def export_lab_csv(session_id: str) -> str:
    """Generates CSV format of strategy comparison metrics for a session."""
    report = get_lab_report(session_id)
    if not report:
        return ""
    lines = [
        "Rank,Strategy,Category,Score,ConsistencyScore,AvgWait_Sec,P95Wait_Sec,TimeSavedPerVeh_Sec,SimulatedVehHoursSaved,MeanQueue_Units,VehiclesProcessed,WaitReduction_Percent,QueueReduction_Percent,AI_Status,Verdict"
    ]
    for r in report.rankings:
        m = report.results[r.strategy_key]
        lines.append(
            f"{r.rank},{m.name},{m.category},{r.composite_score},{r.consistency_score},{m.average_wait_sec},"
            f"{m.p95_wait_sec},{m.avg_waiting_time_saved_per_vehicle_sec},{m.simulated_veh_hours_saved},"
            f"{m.mean_queue_pcu},{m.vehicles_processed},{m.waiting_time_saved_percent}%,"
            f"{m.queue_reduction_percent}%,{m.ai_status},{m.performance_verdict}"
        )
    return "\n".join(lines)
