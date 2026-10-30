"""
Model manager for RL PPO model persistence.

Handles save/load/versioning of trained Stable-Baselines3 models
under backend/models/rl/.

Directory layout:
    backend/models/rl/
        ppo_traffic_latest.zip      # most recent model
        ppo_traffic_v<N>.zip        # versioned checkpoints
        metadata.json               # training run metadata
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default model directory — relative to repo root
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "backend" / "models" / "rl"
LATEST_MODEL_FILENAME = "ppo_traffic_latest"  # SB3 appends .zip automatically


class ModelManager:
    """Manages saving, loading, and versioning of PPO model checkpoints."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.model_dir / "metadata.json"

    # ──────────────────────────────────────────────────────────
    # Paths
    # ──────────────────────────────────────────────────────────

    @property
    def latest_model_path(self) -> Path:
        """Path WITHOUT .zip extension (SB3 convention)."""
        return self.model_dir / LATEST_MODEL_FILENAME

    @property
    def latest_model_zip(self) -> Path:
        """Full path WITH .zip extension."""
        return self.model_dir / (LATEST_MODEL_FILENAME + ".zip")

    def versioned_path(self, version: int) -> Path:
        return self.model_dir / f"ppo_traffic_v{version:04d}"

    # ──────────────────────────────────────────────────────────
    # Existence checks
    # ──────────────────────────────────────────────────────────

    def has_trained_model(self) -> bool:
        """Return True if a trained model file exists."""
        return self.latest_model_zip.exists()

    # ──────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────

    def save(self, model, metadata: Optional[dict] = None) -> Path:
        """
        Save SB3 model as latest and write metadata.

        Args:
            model:    Trained SB3 PPO instance.
            metadata: Optional dict with training statistics to persist.

        Returns:
            Path to the saved .zip file.
        """
        model.save(str(self.latest_model_path))
        logger.info(f"Model saved → {self.latest_model_zip}")

        meta = metadata or {}
        meta["saved_at"] = datetime.utcnow().isoformat()
        meta["model_file"] = self.latest_model_zip.name
        self._write_metadata(meta)
        return self.latest_model_zip

    def save_checkpoint(self, model, timestep: int, metadata: Optional[dict] = None) -> Path:
        """Save a versioned checkpoint alongside the latest model."""
        checkpoint_path = self.model_dir / f"ppo_checkpoint_{timestep}"
        model.save(str(checkpoint_path))
        logger.info(f"Checkpoint saved → {checkpoint_path}.zip")

        # Also update latest
        shutil.copy(str(checkpoint_path) + ".zip", str(self.latest_model_zip))

        meta = metadata or {}
        meta["last_checkpoint_timestep"] = timestep
        meta["last_checkpoint_path"] = checkpoint_path.name + ".zip"
        self._update_metadata(meta)
        return Path(str(checkpoint_path) + ".zip")

    # ──────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────

    def load(self, env=None):
        """
        Load the latest PPO model.

        Args:
            env: Optional environment to attach (for continued training).

        Returns:
            Loaded SB3 PPO model.

        Raises:
            FileNotFoundError: If no trained model exists.
        """
        from stable_baselines3 import PPO

        if not self.has_trained_model():
            raise FileNotFoundError(
                f"No trained model found at {self.latest_model_zip}. "
                "Train a model first using /api/rl/train."
            )
        model = PPO.load(str(self.latest_model_path), env=env)
        logger.info(f"Model loaded ← {self.latest_model_zip}")
        return model

    # ──────────────────────────────────────────────────────────
    # Metadata
    # ──────────────────────────────────────────────────────────

    def get_metadata(self) -> dict:
        if not self._metadata_path.exists():
            return {}
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Could not read metadata: {exc}")
            return {}

    def _write_metadata(self, data: dict):
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _update_metadata(self, updates: dict):
        meta = self.get_metadata()
        meta.update(updates)
        self._write_metadata(meta)

    def list_checkpoints(self) -> list:
        """Return a sorted list of checkpoint .zip files."""
        return sorted(
            str(p) for p in self.model_dir.glob("ppo_checkpoint_*.zip")
        )


# Module-level singleton
model_manager = ModelManager()

