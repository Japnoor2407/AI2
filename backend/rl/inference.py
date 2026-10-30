"""
RL inference module — loads a trained PPO model and runs single-step predictions.

Used by the RL controller and API endpoints to serve real-time signal decisions
without re-instantiating the full environment for every request.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from backend.rl.model_manager import model_manager
from backend.rl.action_space import ActionDecoder

logger = logging.getLogger(__name__)


class RLInferenceEngine:
    """
    Wraps a loaded SB3 PPO model for inference.

    Keeps the model in memory after first load, supporting fast repeated calls.
    Thread-safety note: SB3 predict() is stateless — safe for concurrent use
    as long as the model is not replaced mid-request.
    """

    def __init__(self):
        self._model = None
        self._decoder = ActionDecoder()
        self._load_attempted = False

    # ──────────────────────────────────────────────────────────
    # Model management
    # ──────────────────────────────────────────────────────────

    def load(self, env=None) -> bool:
        """
        Load the latest trained model.

        Args:
            env: Optional environment reference (for continued training).

        Returns:
            True if model loaded successfully, False otherwise.
        """
        self._load_attempted = True
        try:
            self._model = model_manager.load(env=env)
            logger.info("RLInferenceEngine: model loaded successfully.")
            return True
        except FileNotFoundError as exc:
            logger.warning(f"RLInferenceEngine: {exc}")
            return False
        except Exception as exc:
            logger.error(f"RLInferenceEngine: unexpected load error: {exc}")
            return False

    def is_ready(self) -> bool:
        """Return True if a model is loaded and ready for inference."""
        return self._model is not None

    def unload(self):
        """Release model from memory."""
        self._model = None
        logger.info("RLInferenceEngine: model unloaded.")

    # ──────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, int, str]:
        """
        Predict the best action for the given observation vector.

        Args:
            observation:  21-dimensional float32 observation array.
            deterministic: Use argmax policy (True) or sample (False).

        Returns:
            (phase_id, green_duration_sec, description_str)

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "RL model is not loaded. Call load() first or train the model."
            )

        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        action_arr, _ = self._model.predict(obs, deterministic=deterministic)
        action_idx = int(action_arr.item()) if isinstance(action_arr, np.ndarray) else int(action_arr)

        phase_id, duration = self._decoder.decode(action_idx)
        description = self._decoder.describe(action_idx)

        return phase_id, duration, description

    def predict_from_approaches(
        self,
        approaches: dict,
        active_phase_id: int,
        elapsed_in_phase: int,
        current_step: int,
        total_steps: int,
        deterministic: bool = True,
    ) -> Tuple[int, int, str]:
        """
        Convenience wrapper: build observation from approach states, then predict.

        Args:
            approaches:      Dict[str, ApproachState] from JunctionSimulator.
            active_phase_id: Current active phase (1 or 2).
            elapsed_in_phase: Seconds elapsed in current phase.
            current_step:    Simulation step index.
            total_steps:     Total episode duration in steps.
            deterministic:   Argmax policy if True.

        Returns:
            (phase_id, green_duration_sec, description_str)
        """
        from backend.rl.observation import ObservationBuilder
        builder = ObservationBuilder()
        obs = builder.build(
            approaches=approaches,
            active_phase_id=active_phase_id,
            elapsed_in_phase=elapsed_in_phase,
            current_step=current_step,
            total_steps=total_steps,
        )
        return self.predict(obs, deterministic=deterministic)


# Module-level singleton (lazy-loaded on first use)
rl_inference = RLInferenceEngine()

