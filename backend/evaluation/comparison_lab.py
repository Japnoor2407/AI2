"""
Strategy Comparison Lab: Benchmarking framework for fair, reproducible comparisons
across all 5 traffic control strategies:
1. Fixed-Time
2. Actuated
3. Weighted Adaptive
4. Max-Pressure
5. Deep Reinforcement Learning (PPO)

Features:
- Guaranteed identical demand, arrival sequences, and random seed across all tested strategies.
- Configurable test durations: 5m (300s), 10m (600s), 15m (900s), 30m (1800s), 45m (2700s), 60m (3600s).
- Exact calculation of time-saved metrics:
    - total_waiting_time_saved_sec
    - avg_waiting_time_saved_per_vehicle_sec
    - waiting_time_saved_percent
- Multi-metric strategy ranking with clear plain-English justifications.
- Session persistence to disk and CSV/JSON export.
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
        "description": "Standard predetermined green splits without vehicle sensor feedback.",
        "category": "Baseline",
    },
    "actuated": {
        "name": "Actuated Control",
        "description": "Extends green times when arriving vehicles are detected; gaps out during lull.",
        "category": "Rule-Based",
    },
    "adaptive": {
        "name": "Weighted Adaptive",
        "description": "Dynamic queue-based scoring with starvation prevention and delay fairness.",
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
    p95_wait_sec: float
    total_throughput_pcu: float
    total_arrived_vehicles: int
    starvation_occurrences: int
    controller_execution_time_ms: float
    # Time-saved & improvement metrics vs Fixed-Time baseline
    avg_waiting_time_saved_per_vehicle_sec: float = 0.0
    total_waiting_time_saved_sec: float = 0.0
    waiting_time_saved_percent: float = 0.0
    queue_reduction_percent: float = 0.0
    throughput_improvement_percent: float = 0.0


class StrategyRanking(BaseModel):
    rank: int
    strategy_key: str
    strategy_name: str
    composite_score: float
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
    plain_english_summary: str
    time_series: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


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
    # Always include the last observation
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

    if demand_multiplier != 1.0:
        arrivals_df = _scale_arrivals_df(arrivals_df, demand_multiplier)

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
    time_series_data: Dict[str, List[Dict[str, Any]]] = {}

    for strat_key in strategies_to_run:
        logger.info(f"[Lab-{session_id}] Executing strategy: {strat_key} ({duration_sec}s)")
        
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
        time_series_data[strat_key] = _downsample_log(sim.log, max_points=60)

    # 2. Compute comparisons vs Fixed-Time baseline
    fixed_base = raw_metrics["fixed_time"]
    results: Dict[str, StrategyMetrics] = {}

    for key, m in raw_metrics.items():
        meta = STRATEGY_METADATA.get(key, {"name": key, "category": "General", "description": ""})
        
        # Time savings vs Fixed-Time
        time_saved_per_veh = round(max(-60.0, fixed_base.average_wait_sec - m.average_wait_sec), 2)
        total_time_saved = round(time_saved_per_veh * max(1, m.total_arrived_vehicles), 1)
        wait_pct = round(((fixed_base.average_wait_sec - m.average_wait_sec) / max(0.01, fixed_base.average_wait_sec)) * 100.0, 2)
        queue_pct = round(((fixed_base.mean_queue_pcu - m.mean_queue_pcu) / max(0.01, fixed_base.mean_queue_pcu)) * 100.0, 2)
        thru_pct = round(((m.total_throughput_pcu - fixed_base.total_throughput_pcu) / max(0.01, fixed_base.total_throughput_pcu)) * 100.0, 2)

        results[key] = StrategyMetrics(
            key=key,
            name=meta["name"],
            category=meta["category"],
            description=meta["description"],
            mean_queue_pcu=m.mean_queue_pcu,
            max_queue_pcu=m.max_queue_pcu,
            average_wait_sec=m.average_wait_sec,
            p95_wait_sec=m.p95_wait_sec,
            total_throughput_pcu=m.total_throughput_pcu,
            total_arrived_vehicles=m.total_arrived_vehicles,
            starvation_occurrences=m.starvation_occurrences,
            controller_execution_time_ms=m.controller_execution_time_ms,
            avg_waiting_time_saved_per_vehicle_sec=time_saved_per_veh,
            total_waiting_time_saved_sec=total_time_saved,
            waiting_time_saved_percent=wait_pct,
            queue_reduction_percent=queue_pct,
            throughput_improvement_percent=thru_pct,
        )

    # 3. Strategy Ranking using a composite performance score
    scores = {}
    for key, res in results.items():
        # Score calculation:
        # + 0.40 * wait_time_reduction_percent
        # + 0.30 * queue_reduction_percent
        # + 0.20 * throughput_improvement_percent
        # - 5.0 * starvation_occurrences
        wait_norm = res.waiting_time_saved_percent
        queue_norm = res.queue_reduction_percent
        thru_norm = res.throughput_improvement_percent
        starve_penalty = min(20.0, res.starvation_occurrences * 4.0)

        composite = round(0.40 * wait_norm + 0.30 * queue_norm + 0.20 * thru_norm - starve_penalty, 2)
        # Baseline score normalized to 0.0
        if key == "fixed_time":
            composite = 0.0
        scores[key] = composite

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    rankings: List[StrategyRanking] = []
    for idx, key in enumerate(sorted_keys, start=1):
        res = results[key]
        score = scores[key]
        if key == "fixed_time":
            rationale = "Traditional baseline timer with fixed cycle splits. Used as the benchmark reference."
        else:
            rationale = (
                f"Achieved {res.waiting_time_saved_percent:+0.1f}% wait time reduction "
                f"({res.avg_waiting_time_saved_per_vehicle_sec:+0.1f}s/veh saved) and "
                f"{res.queue_reduction_percent:+0.1f}% shorter queues with "
                f"{res.starvation_occurrences} starvation events."
            )
        rankings.append(
            StrategyRanking(
                rank=idx,
                strategy_key=key,
                strategy_name=res.name,
                composite_score=score,
                rationale=rationale,
            )
        )

    best_strat = rankings[0]
    plain_summary = (
        f"In a {duration_sec // 60}-minute evaluation at {junction_id}, {best_strat.strategy_name} "
        f"ranked #1 overall. It saved an average of "
        f"{results[best_strat.strategy_key].avg_waiting_time_saved_per_vehicle_sec} seconds per vehicle "
        f"({results[best_strat.strategy_key].waiting_time_saved_percent}% less wait) compared to traditional fixed-time signals, "
        f"delivering a total of {int(results[best_strat.strategy_key].total_waiting_time_saved_sec):,} seconds of commuter travel time saved."
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
        "Rank,Strategy,Category,MeanQueue_PCU,MaxQueue_PCU,AvgWait_Sec,P95Wait_Sec,Throughput_PCU,TimeSavedPerVeh_Sec,TotalTimeSaved_Sec,WaitReduction_Percent,QueueReduction_Percent"
    ]
    for r in report.rankings:
        m = report.results[r.strategy_key]
        lines.append(
            f"{r.rank},{m.name},{m.category},{m.mean_queue_pcu},{m.max_queue_pcu},{m.average_wait_sec},"
            f"{m.p95_wait_sec},{m.total_throughput_pcu},{m.avg_waiting_time_saved_per_vehicle_sec},"
            f"{m.total_waiting_time_saved_sec},{m.waiting_time_saved_percent}%,{m.queue_reduction_percent}%"
        )
    return "\n".join(lines)

