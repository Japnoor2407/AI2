"""
FastAPI REST API — all endpoints for the Adaptive Traffic Signal Optimization Agent.
"""

import asyncio
import csv
import io
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.core.schema import ControllerMode, EvaluationReport
from backend.core.state import app_state
from backend.evaluation.evaluator import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SIH1607 Adaptive Traffic Signal API",
    description="AI-based adaptive traffic signal optimization — Pune CCTV dataset",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# Request / Response bodies
# ──────────────────────────────────────────────────────────

class InitRequest(BaseModel):
    junction_id: str = "AlankarChowk"
    date: str = "11"
    duration_sec: int = 1800
    mode: str = "ADAPTIVE"


class ModeRequest(BaseModel):
    mode: str  # "FIXED_TIME", "ACTUATED", "ADAPTIVE", or "PREDICTION_ASSISTED"


class SeekRequest(BaseModel):
    target_step: int


class EvalRequest(BaseModel):
    junction_id: str = "AlankarChowk"
    date: str = "11"
    duration_sec: int = 900
    scenario: str = "normal"
    demand_multiplier: float = 1.0


# ──────────────────────────────────────────────────────────
# Startup: auto-initialize with defaults
# ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    try:
        app_state.initialize(
            junction_id="AlankarChowk",
            date="11",
            duration_sec=1800,
            mode=ControllerMode.ADAPTIVE,
        )
        logger.info("Default simulation initialized on startup.")
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}")


# ──────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "initialized": app_state._initialized,
        "junction": app_state.junction_id,
        "date": app_state.date,
        "mode": app_state.mode.value if app_state.mode else None,
        "step": app_state.sim.step_count if app_state.sim else 0,
        "total_steps": app_state.sim.total_steps if app_state.sim else 0,
    }


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────

@app.get("/api/dataset/summary")
def dataset_summary():
    return app_state.adapter.get_dataset_summary()


@app.get("/api/dataset/schema")
def dataset_schema():
    return {
        "columns": [
            {"name": "Time", "type": "str (HH:MM:SS)", "description": "Timestamp of observation interval"},
            {"name": "Direction", "type": "str", "allowed_values": ["UP", "DOWN", "LEFT", "RIGHT"],
             "description": "Directional movement through camera"},
            {"name": "car", "type": "int (cumulative)", "pcu_weight": 1.0, "description": "Cars detected (cumulative counter)"},
            {"name": "motorbike", "type": "int (cumulative)", "pcu_weight": 0.5, "description": "Motorbikes detected"},
            {"name": "bus", "type": "int (cumulative)", "pcu_weight": 2.5, "description": "Buses detected"},
            {"name": "truck", "type": "int (cumulative)", "pcu_weight": 2.5, "description": "Trucks detected"},
        ],
        "granularity": "25 frames per second (25 rows per direction per second)",
        "count_semantics": "CUMULATIVE — use delta(t) = count(t) - count(t-1) to get arrivals",
        "pcu_standard": "IRC:106 Indian Urban Roads",
        "junctions": app_state.adapter.get_available_junctions(),
    }


# ──────────────────────────────────────────────────────────
# Traffic (live replay state)
# ──────────────────────────────────────────────────────────

@app.get("/api/traffic/current")
def traffic_current():
    if app_state.last_observation is None:
        return {
            "status": "not_started",
            "message": "Simulation not started. POST /api/simulation/start first.",
        }
    obs = app_state.last_observation
    return {
        "step": obs.step,
        "timestamp": obs.timestamp_str,
        "total_queue_pcu": obs.total_active_queue_pcu,
        "total_throughput_pcu": obs.total_throughput_pcu,
        "average_delay_sec": obs.average_delay_sec,
        "approaches": {
            app_id: {
                "name": a.approach_name,
                "direction": a.direction,
                "signal": a.current_signal.value,
                "queue_pcu": a.estimated_queue_pcu,
                "queue_vehicles": a.estimated_queue_vehicles,
                "arrival_rate_pcu_per_min": a.arrival_rate_pcu_per_min,
                "avg_wait_sec": a.average_wait_time_sec,
                "time_since_served_sec": a.time_since_last_served_sec,
                "is_starving": a.is_starving,
                "arrivals": {
                    "car": a.arrivals_last_step.car,
                    "motorbike": a.arrivals_last_step.motorbike,
                    "bus": a.arrivals_last_step.bus,
                    "truck": a.arrivals_last_step.truck,
                },
            }
            for app_id, a in obs.approaches.items()
        },
        "controller": app_state.last_decision.model_dump() if app_state.last_decision else None,
    }


@app.get("/api/traffic/history")
def traffic_history(limit: int = Query(default=300, ge=1, le=5000)):
    recent = app_state.history[-limit:]
    return {
        "count": len(recent),
        "history": [
            {
                "step": obs.step,
                "time": obs.timestamp_str,
                "queue_pcu": obs.total_active_queue_pcu,
                "throughput_pcu": obs.total_throughput_pcu,
                "delay_sec": obs.average_delay_sec,
                "phase_id": obs.controller_decision.active_phase_id,
                "signal_state": obs.controller_decision.signal_state.value,
            }
            for obs in recent
        ],
    }


@app.get("/api/traffic/aggregated")
def traffic_aggregated(
    interval_sec: int = Query(default=600, ge=60, le=3600),
    junction_id: Optional[str] = None,
    date: Optional[str] = None,
    camera: Optional[str] = None,
):
    """
    Returns traffic analytics aggregated into 10-minute (or custom interval) units.
    Includes vehicle class volumes, PCU arrivals, flow rates, and simulated queues/delays.
    """
    j = junction_id or app_state.junction_id
    d = date or app_state.date
    if not camera:
        cams = app_state.adapter.get_cameras_for_junction(j)
        c = cams[0] if cams else "a2"
    else:
        c = camera

    intervals = app_state.adapter.load_aggregated_intervals(
        junction=j, camera=c, date=d, interval_sec=interval_sec, max_intervals=18
    )

    # Compute simulation performance per interval if history exists
    hist = app_state.history
    sim_bins = []
    if hist:
        bin_size = interval_sec
        total_steps = len(hist)
        num_bins = max(1, (total_steps + bin_size - 1) // bin_size)
        for b in range(num_bins):
            chunk = hist[b * bin_size : (b + 1) * bin_size]
            if not chunk:
                continue
            q_vals = [obs.total_active_queue_pcu for obs in chunk]
            d_vals = [obs.average_delay_sec for obs in chunk]
            t_start = chunk[0].timestamp_str
            t_end = chunk[-1].timestamp_str
            sim_bins.append({
                "interval_index": b + 1,
                "label": f"{t_start[:5]}-{t_end[:5]}",
                "start_time": t_start,
                "end_time": t_end,
                "mean_queue_pcu": round(sum(q_vals) / len(q_vals), 2),
                "peak_queue_pcu": round(max(q_vals), 2),
                "mean_delay_sec": round(sum(d_vals) / len(d_vals), 2),
                "throughput_pcu": round(chunk[-1].total_throughput_pcu - chunk[0].total_throughput_pcu, 2),
            })

    return {
        "status": "ok",
        "junction": j,
        "date": d,
        "camera": c,
        "unit": f"{interval_sec // 60}min",
        "interval_seconds": interval_sec,
        "intervals": intervals,
        "simulation_intervals": sim_bins,
    }


# ──────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────

@app.get("/api/controller/status")
def controller_status():
    return {
        "mode": app_state.mode.value,
        "is_running": app_state.is_running,
        "is_paused": app_state.is_paused,
        "step": app_state.sim.step_count if app_state.sim else 0,
        "total_steps": app_state.sim.total_steps if app_state.sim else 0,
        "junction": app_state.junction_id,
        "date": app_state.date,
        "current_phase": (
            app_state.last_decision.active_phase_id if app_state.last_decision else None
        ),
        "signal_state": (
            app_state.last_decision.signal_state.value if app_state.last_decision else None
        ),
        "decision_rationale": (
            app_state.last_decision.decision_rationale if app_state.last_decision else None
        ),
    }


@app.post("/api/controller/mode")
def set_mode(req: ModeRequest):
    try:
        new_mode = ControllerMode(req.mode)
    except ValueError:
        raise HTTPException(400, f"Invalid mode '{req.mode}'. Use FIXED_TIME, ACTUATED, ADAPTIVE, or PREDICTION_ASSISTED.")
    app_state.mode = new_mode
    return {"status": "ok", "mode": new_mode.value}


@app.post("/api/controller/step")
def controller_step():
    """Advance simulation by exactly one second."""
    if not app_state._initialized:
        raise HTTPException(400, "Simulation not initialized.")
    obs = app_state.step_once()
    if obs is None:
        return {"status": "finished", "message": "Dataset replay complete."}
    return {
        "status": "ok",
        "step": obs.step,
        "time": obs.timestamp_str,
        "queue_pcu": obs.total_active_queue_pcu,
        "phase": app_state.last_decision.phase_name if app_state.last_decision else None,
        "signal": app_state.last_decision.signal_state.value if app_state.last_decision else None,
        "rationale": app_state.last_decision.decision_rationale if app_state.last_decision else None,
    }


# ──────────────────────────────────────────────────────────
# Simulation control
# ──────────────────────────────────────────────────────────

@app.post("/api/simulation/start")
async def simulation_start(req: Optional[InitRequest] = None):
    if req:
        try:
            mode = ControllerMode(req.mode)
        except ValueError:
            raise HTTPException(400, f"Invalid mode: {req.mode}")
        app_state.initialize(
            junction_id=req.junction_id,
            date=req.date,
            duration_sec=req.duration_sec,
            mode=mode,
        )
    app_state.is_running = True
    app_state.is_paused = False
    return {"status": "started", "junction": app_state.junction_id, "mode": app_state.mode.value}


@app.post("/api/simulation/pause")
def simulation_pause():
    app_state.is_paused = not app_state.is_paused
    return {"status": "paused" if app_state.is_paused else "resumed"}


@app.post("/api/simulation/reset")
def simulation_reset():
    app_state.reset()
    return {"status": "reset"}


@app.post("/api/simulation/seek")
def simulation_seek(req: SeekRequest):
    """Fast-forwards or rewinds replay simulation to target_step."""
    if not app_state._initialized:
        app_state.initialize(
            junction_id="AlankarChowk",
            date="11",
            duration_sec=1800,
            mode=ControllerMode.ADAPTIVE,
        )
    obs = app_state.seek_to_step(req.target_step)
    if obs is None:
        return {"status": "finished", "message": "Seek reached end of dataset."}
    return {
        "status": "ok",
        "step": obs.step,
        "time": obs.timestamp_str,
        "total_queue_pcu": obs.total_active_queue_pcu,
        "total_throughput_pcu": obs.total_throughput_pcu,
        "average_delay_sec": obs.average_delay_sec,
        "active_phase_id": obs.controller_decision.active_phase_id,
        "phase_name": obs.controller_decision.phase_name,
        "signal_state": obs.controller_decision.signal_state.value,
        "rationale": obs.controller_decision.decision_rationale,
    }


# ──────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────

@app.get("/api/evaluation/summary")
def evaluation_summary():
    if app_state.evaluation_report is None:
        return {
            "status": "not_evaluated",
            "message": "No evaluation run yet. POST /api/evaluation/run to evaluate.",
        }
    r = app_state.evaluation_report
    return {
        "status": "ok",
        "scenario_id": r.scenario_id,
        "junction": r.junction_name,
        "duration_sec": r.evaluation_duration_sec,
        "fixed_time": {
            "mean_queue_pcu": r.fixed_time_baseline.mean_queue_pcu,
            "max_queue_pcu": r.fixed_time_baseline.max_queue_pcu,
            "average_wait_sec": r.fixed_time_baseline.average_wait_sec,
            "total_throughput_pcu": r.fixed_time_baseline.total_throughput_pcu,
            "starvation_occurrences": r.fixed_time_baseline.starvation_occurrences,
        },
        "adaptive": {
            "mean_queue_pcu": r.adaptive_controller.mean_queue_pcu,
            "max_queue_pcu": r.adaptive_controller.max_queue_pcu,
            "average_wait_sec": r.adaptive_controller.average_wait_sec,
            "total_throughput_pcu": r.adaptive_controller.total_throughput_pcu,
            "starvation_occurrences": r.adaptive_controller.starvation_occurrences,
        },
        "improvements": {
            "queue_reduction_percent": r.queue_reduction_percent,
            "waiting_time_reduction_percent": r.waiting_time_reduction_percent,
            "throughput_improvement_percent": r.throughput_improvement_percent,
        },
        "summary": r.summary_text,
        "evaluated_at": r.timestamp_evaluated,
        "time_series_length": len(r.time_series),
    }


@app.post("/api/evaluation/run")
def evaluation_run(req: Optional[EvalRequest] = None):
    j = req.junction_id if req else "AlankarChowk"
    d = req.date if req else "11"
    dur = req.duration_sec if req else 900
    scenario = req.scenario if req else "normal"
    demand_mult = req.demand_multiplier if req else 1.0

    try:
        report = run_evaluation(
            junction_id=j, date=d, duration_sec=dur,
            scenario=scenario, demand_multiplier=demand_mult,
        )
        app_state.evaluation_report = report
        return {"status": "ok", "summary": report.summary_text}
    except FileNotFoundError as e:
        raise HTTPException(404, f"Dataset file not found: {e}")
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(500, f"Evaluation error: {e}")


@app.post("/api/evaluation/corridor")
def evaluation_corridor(
    date: str = "11",
    duration_sec: int = Query(default=300, ge=60, le=1800),
):
    """Executes a comparative benchmark of Independent vs Coordinated operation across the corridor."""
    try:
        from backend.simulation.corridor import run_corridor_evaluation
        result = run_corridor_evaluation(date=date, duration_sec=duration_sec)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, f"Corridor dataset file not found: {e}")
    except Exception as e:
        logger.exception("Corridor evaluation failed")
        raise HTTPException(500, f"Corridor evaluation error: {e}")


@app.get("/api/evaluation/export")
def evaluation_export(format: str = Query(default="json", pattern="^(json|csv)$")):
    if app_state.evaluation_report is None:
        raise HTTPException(404, "No evaluation results. Run /api/evaluation/run first.")

    r = app_state.evaluation_report

    if format == "json":
        content = r.model_dump()
        return JSONResponse(content=content, headers={
            "Content-Disposition": "attachment; filename=evaluation_results.json"
        })
    else:
        # CSV export of time series
        ts = r.time_series
        if not ts:
            raise HTTPException(404, "No time-series data to export.")
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=ts[0].keys())
        writer.writeheader()
        writer.writerows(ts)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=evaluation_timeseries.csv"},
        )


@app.get("/api/decision-log")
def decision_log(limit: int = Query(default=100, ge=1, le=2000)):
    recent = app_state.decision_log[-limit:]
    return {
        "count": len(recent),
        "log": [
            {
                "step": d.step,
                "time": d.timestamp_str,
                "mode": d.mode.value,
                "phase_id": d.active_phase_id,
                "phase_name": d.phase_name,
                "signal": d.signal_state.value,
                "time_remaining": d.time_remaining_in_state,
                "demand_scores": d.phase_demand_scores,
                "rationale": d.decision_rationale,
                "is_transitioning": d.is_transitioning,
            }
            for d in recent
        ],
    }


# ──────────────────────────────────────────────────────────
# Safety Audit Log
# ──────────────────────────────────────────────────────────

@app.get("/api/safety/events")
def safety_events(limit: int = Query(default=200, ge=1, le=5000)):
    """Return all safety supervisor override events for the current session."""
    events = app_state.safety_events[-limit:]
    return {
        "count": len(events),
        "total_violations_prevented": (
            app_state.safety_supervisor.total_violations_prevented
            if app_state.safety_supervisor else 0
        ),
        "events": [e.model_dump() for e in events],
    }


# ──────────────────────────────────────────────────────────
# Controller Modes
# ──────────────────────────────────────────────────────────

@app.get("/api/controller/modes")
def controller_modes():
    """List all available controller types with descriptions."""
    return {
        "modes": [
            {
                "value": "FIXED_TIME",
                "label": "Fixed-Time",
                "description": "Predetermined cycle with fixed green splits. Baseline controller.",
            },
            {
                "value": "ACTUATED",
                "label": "Actuated",
                "description": "Gap-out / max-out demand-responsive controller. Extends green based on vehicle arrivals.",
            },
            {
                "value": "ADAPTIVE",
                "label": "Adaptive (AI)",
                "description": "Demand-scored adaptive controller with starvation prevention and anti-congestion override.",
            },
            {
                "value": "PREDICTION_ASSISTED",
                "label": "Prediction-Assisted (ML)",
                "description": "Short-term ML demand forecasting integrated with proactive green duration allocation.",
            },
        ]
    }


# ──────────────────────────────────────────────────────────
# Replay History
# ──────────────────────────────────────────────────────────

@app.get("/api/replay/history")
def replay_history(limit: int = Query(default=500, ge=1, le=10000)):
    """Return full step-by-step replay log for timeline visualization."""
    recent = app_state.history[-limit:]
    return {
        "count": len(recent),
        "history": [
            {
                "step": obs.step,
                "time": obs.timestamp_str,
                "phase_id": obs.controller_decision.active_phase_id,
                "phase_name": obs.controller_decision.phase_name,
                "signal_state": obs.controller_decision.signal_state.value,
                "total_queue_pcu": obs.total_active_queue_pcu,
                "total_throughput_pcu": obs.total_throughput_pcu,
                "average_delay_sec": obs.average_delay_sec,
                "is_transitioning": obs.controller_decision.is_transitioning,
                "safety_override": getattr(obs.controller_decision, "safety_override", False),
                "approaches": {
                    app_id: {
                        "signal": a.current_signal.value,
                        "queue_pcu": a.estimated_queue_pcu,
                        "wait_sec": a.average_wait_time_sec,
                        "is_starving": a.is_starving,
                    }
                    for app_id, a in obs.approaches.items()
                },
            }
            for obs in recent
        ],
    }


# ──────────────────────────────────────────────────────────
# ML Forecaster
# ──────────────────────────────────────────────────────────

@app.post("/api/ml/train")
def ml_train():
    """Train the traffic demand forecaster on the current junction dataset."""
    try:
        from backend.models.forecaster import TrafficDemandForecaster
        from backend.data.loader import DatasetAdapter

        adapter = DatasetAdapter()
        junction = app_state.junction_id
        date = app_state.date
        cameras = adapter.get_cameras_for_junction(junction)
        if not cameras:
            raise HTTPException(404, f"No cameras found for junction {junction}")

        cam = cameras[0]
        df = adapter.load_camera_arrivals(junction, cam, date, max_seconds=7200)
        forecaster = TrafficDemandForecaster()
        metrics = forecaster.train_and_evaluate(df)

        # Cache on app_state for later prediction
        app_state._ml_forecaster = forecaster
        return {"status": "ok", "metrics": metrics}
    except FileNotFoundError as e:
        raise HTTPException(404, f"Dataset not found: {e}")
    except Exception as e:
        logger.exception("ML training failed")
        raise HTTPException(500, f"ML training error: {e}")


@app.get("/api/ml/metrics")
def ml_metrics():
    """Return metrics from the last ML training run."""
    forecaster = getattr(app_state, "_ml_forecaster", None)
    if forecaster is None:
        return {"status": "not_trained", "message": "POST /api/ml/train first."}
    return {"status": "ok", "metrics": forecaster.metrics, "model": forecaster.model_name}


@app.get("/api/ml/predict")
def ml_predict(horizon_steps: int = Query(default=30, ge=1, le=300)):
    """Return short-term traffic demand prediction for the next N steps."""
    forecaster = getattr(app_state, "_ml_forecaster", None)
    if forecaster is None:
        return {"status": "not_trained", "message": "POST /api/ml/train first."}
    if not app_state.history:
        return {"status": "no_data", "message": "Start simulation first to collect history."}
    try:
        predictions = forecaster.predict_next(
            history=app_state.history[-60:],
            horizon=horizon_steps,
        )
        return {"status": "ok", "horizon_steps": horizon_steps, "predictions": predictions}
    except Exception as e:
        logger.exception("ML prediction failed")
        raise HTTPException(500, f"Prediction error: {e}")


# ──────────────────────────────────────────────────────────
# Evaluation — Full 3-controller summary endpoint
# ──────────────────────────────────────────────────────────

@app.get("/api/evaluation/full-summary")
def evaluation_full_summary():
    """Return 3-controller comparison (Fixed, Actuated, Adaptive) from last evaluation."""
    if app_state.evaluation_report is None:
        return {
            "status": "not_evaluated",
            "message": "No evaluation run yet. POST /api/evaluation/run to evaluate.",
        }
    r = app_state.evaluation_report

    def _fmt(m):
        return {
            "mean_queue_pcu": m.mean_queue_pcu,
            "max_queue_pcu": m.max_queue_pcu,
            "average_wait_sec": m.average_wait_sec,
            "median_wait_sec": m.median_wait_sec,
            "p95_wait_sec": m.p95_wait_sec,
            "total_throughput_pcu": m.total_throughput_pcu,
            "residual_queue_pcu": m.residual_queue_pcu,
            "starvation_occurrences": m.starvation_occurrences,
            "execution_time_ms": m.controller_execution_time_ms,
        }

    return {
        "status": "ok",
        "scenario_id": r.scenario_id,
        "scenario_name": r.scenario_name,
        "junction": r.junction_name,
        "duration_sec": r.evaluation_duration_sec,
        "evaluated_at": r.timestamp_evaluated,
        "fixed_time": _fmt(r.fixed_time_baseline),
        "actuated": _fmt(r.actuated_baseline) if r.actuated_baseline else None,
        "adaptive": _fmt(r.adaptive_controller),
        "improvements_vs_fixed": {
            "queue_reduction_percent": r.queue_reduction_percent,
            "waiting_time_reduction_percent": r.waiting_time_reduction_percent,
            "throughput_improvement_percent": r.throughput_improvement_percent,
        },
        "improvements_vs_actuated": r.improvements_vs_actuated,
        "summary": r.summary_text,
        "time_series_length": len(r.time_series),
    }

