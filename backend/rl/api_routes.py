"""
FastAPI router for RL (Deep Reinforcement Learning) endpoints.

Endpoints:
  GET  /api/rl/status          — model availability and metadata
  POST /api/rl/train           — start background training
  GET  /api/rl/training/status — training job progress
  POST /api/rl/predict         — single-step action prediction
  POST /api/rl/evaluate        — run comparative evaluation (Fixed / Adaptive / PPO)
  GET  /api/rl/metrics         — training metrics history
  GET  /api/rl/comparison      — latest benchmark comparison table
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.rl.model_manager import model_manager
from backend.rl.inference import rl_inference
from backend.rl.sumo_adapter import is_sumo_available

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rl", tags=["Deep RL"])

# ──────────────────────────────────────────────────────────
# In-memory training job state
# ──────────────────────────────────────────────────────────

class _TrainingJob:
    def __init__(self):
        self.is_running: bool = False
        self.timesteps_done: int = 0
        self.total_timesteps: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: str = "idle"  # idle | running | done | error
        self.error_message: Optional[str] = None
        self.mean_ep_reward: Optional[float] = None


_training_job = _TrainingJob()


# ──────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    total_timesteps: int = 50_000
    episode_duration_sec: int = 900
    n_envs: int = 1
    checkpoint_freq: int = 10_000
    seed: int = 42
    resume: bool = False
    simulator: str = "auto"  # "auto" | "junction_sim" | "sumo"
    sumo_gui: bool = False


class PredictRequest(BaseModel):
    observation: List[float]
    deterministic: bool = True


class EvaluateRequest(BaseModel):
    duration_sec: int = 900
    seed: int = 0


# ──────────────────────────────────────────────────────────
# Background training task
# ──────────────────────────────────────────────────────────

def _run_training(req: TrainRequest):
    """Runs in a background thread (launched by FastAPI BackgroundTasks)."""
    global _training_job
    _training_job.is_running = True
    _training_job.status = "running"
    _training_job.start_time = time.time()
    _training_job.total_timesteps = req.total_timesteps
    _training_job.error_message = None

    try:
        from backend.rl.train import train
        model = train(
            total_timesteps=req.total_timesteps,
            episode_duration_sec=req.episode_duration_sec,
            n_envs=req.n_envs,
            checkpoint_freq=req.checkpoint_freq,
            seed=req.seed,
            resume=req.resume,
            simulator=req.simulator,
            sumo_gui=req.sumo_gui,
        )
        meta = model_manager.get_metadata()
        _training_job.mean_ep_reward = meta.get("mean_ep_reward_last20")
        _training_job.timesteps_done = req.total_timesteps
        _training_job.status = "done"
        logger.info("Background training completed successfully.")

        # Reload the freshly trained model into the inference engine
        rl_inference.load()

    except Exception as exc:
        logger.error(f"Background training failed: {exc}", exc_info=True)
        _training_job.status = "error"
        _training_job.error_message = str(exc)
    finally:
        _training_job.is_running = False
        _training_job.end_time = time.time()


# ──────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────

@router.get("/status")
def rl_status() -> Dict[str, Any]:
    """Return model availability, training state, metadata, and SUMO status."""
    meta = model_manager.get_metadata()
    return {
        "model_available": model_manager.has_trained_model(),
        "model_loaded": rl_inference.is_ready(),
        "model_path": str(model_manager.latest_model_zip) if model_manager.has_trained_model() else None,
        "sumo_available": is_sumo_available(),
        "training_status": _training_job.status,
        "metadata": meta,
    }


@router.post("/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Start a PPO training job in the background.
    Returns immediately; poll /api/rl/training/status for progress.
    """
    global _training_job
    if _training_job.is_running:
        raise HTTPException(
            status_code=409,
            detail="A training job is already running. Wait for it to complete.",
        )
    _training_job = _TrainingJob()
    _training_job.total_timesteps = req.total_timesteps
    background_tasks.add_task(_run_training, req)
    return {
        "message": "Training started in background.",
        "total_timesteps": req.total_timesteps,
        "episode_duration_sec": req.episode_duration_sec,
        "simulator": req.simulator,
        "seed": req.seed,
    }


@router.get("/training/status")
def training_status() -> Dict[str, Any]:
    """Return current training job progress."""
    elapsed = None
    if _training_job.start_time:
        end = _training_job.end_time or time.time()
        elapsed = round(end - _training_job.start_time, 1)

    progress_pct = 0.0
    if _training_job.total_timesteps > 0:
        progress_pct = round(
            100.0 * _training_job.timesteps_done / _training_job.total_timesteps, 1
        )

    return {
        "status": _training_job.status,
        "is_running": _training_job.is_running,
        "timesteps_done": _training_job.timesteps_done,
        "total_timesteps": _training_job.total_timesteps,
        "progress_percent": progress_pct,
        "elapsed_sec": elapsed,
        "mean_ep_reward": _training_job.mean_ep_reward,
        "error": _training_job.error_message,
    }


@router.post("/predict")
def predict_action(req: PredictRequest) -> Dict[str, Any]:
    """
    Run a single PPO inference step.

    Body: { "observation": [float × 21], "deterministic": true }

    Returns the decoded phase ID, green duration, and description.
    """
    import numpy as np

    if not rl_inference.is_ready():
        # Try loading lazily
        if not rl_inference.load():
            raise HTTPException(
                status_code=503,
                detail="No trained RL model available. Train a model first using POST /api/rl/train.",
            )

    obs = req.observation
    if len(obs) != 21:
        raise HTTPException(
            status_code=422,
            detail=f"Observation must have exactly 21 elements, got {len(obs)}.",
        )

    try:
        phase_id, duration, description = rl_inference.predict(
            np.array(obs, dtype=np.float32),
            deterministic=req.deterministic,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return {
        "phase_id": phase_id,
        "green_duration_sec": duration,
        "description": description,
        "mode": "DEEP_RL",
    }


@router.get("/metrics")
def rl_metrics() -> Dict[str, Any]:
    """Return training metadata and checkpoint list."""
    meta = model_manager.get_metadata()
    checkpoints = model_manager.list_checkpoints()
    return {
        "metadata": meta,
        "checkpoints": checkpoints,
        "model_available": model_manager.has_trained_model(),
    }


@router.post("/evaluate")
def evaluate_with_rl(req: EvaluateRequest) -> Dict[str, Any]:
    """
    Run a comparative evaluation: Fixed-Time vs Adaptive vs PPO.

    Only includes PPO column when a trained model is available.
    Returns summary metrics for all three controllers.
    """
    from backend.evaluation.evaluator import run_evaluation
    from backend.core.state import app_state

    if not app_state._initialized:
        raise HTTPException(
            status_code=400,
            detail="Simulation not initialized. Call POST /api/simulation/init first.",
        )

    try:
        report = run_evaluation(
            junction_id=app_state.junction_id,
            date=app_state.date,
            duration_sec=req.duration_sec,
            seed=req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {exc}")

    result: Dict[str, Any] = {
        "fixed_time": report.fixed_time_baseline.model_dump() if report.fixed_time_baseline else None,
        "adaptive": report.adaptive_controller.model_dump() if report.adaptive_controller else None,
        "queue_reduction_percent": report.queue_reduction_percent,
        "waiting_time_reduction_percent": report.waiting_time_reduction_percent,
        "throughput_improvement_percent": report.throughput_improvement_percent,
        "rl_available": model_manager.has_trained_model(),
        "rl_note": (
            "PPO model is trained and available — use /api/rl/predict for live inference."
            if model_manager.has_trained_model()
            else "No PPO model trained yet. Use POST /api/rl/train to train one."
        ),
    }
    return result


@router.get("/comparison")
def rl_comparison() -> Dict[str, Any]:
    """Return the latest benchmark comparison including RL status."""
    from backend.core.state import app_state

    meta = model_manager.get_metadata()
    return {
        "model_available": model_manager.has_trained_model(),
        "training_metadata": meta,
        "training_status": _training_job.status,
        "mean_ep_reward": _training_job.mean_ep_reward or meta.get("mean_ep_reward_last20"),
        "last_evaluation": (
            app_state.evaluation_report.model_dump()
            if app_state.evaluation_report
            else None
        ),
    }

