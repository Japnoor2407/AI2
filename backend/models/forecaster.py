"""
Short-Term Traffic Demand Forecaster (Phase 3 Requirement).

Predicts short-term vehicle arrivals (PCU) for intersection movements
over the next 30-60 seconds. Uses strict chronological train/val/test splits
to prevent data leakage, and benchmarks ML (Gradient Boosting / Random Forest)
against persistence and moving average baselines.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_PATH = MODEL_DIR / "traffic_forecaster.joblib"


class TrafficDemandForecaster:
    """
    Supervised short-term traffic volume predictor with chronological splits.
    """

    def __init__(self, horizon_steps: int = 30):
        self.horizon_steps = horizon_steps  # e.g., 30 seconds ahead
        self.model: Optional[GradientBoostingRegressor] = None
        self.is_trained: bool = False
        self.metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def _prepare_time_series_features(
        self, df_arrivals: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Builds lagged features chronologically per direction.
        Prevents any future target leakage.
        """
        if (
            df_arrivals is None
            or df_arrivals.empty
            or "Direction" not in df_arrivals.columns
            or "pcu_arrivals" not in df_arrivals.columns
        ):
            return pd.DataFrame(), pd.Series()

        records = []
        for direction, group in df_arrivals.groupby("Direction"):
            g = group.sort_values("Time").copy().reset_index(drop=True)
            pcu = g["pcu_arrivals"].values
            n = len(pcu)
            if n < 10:
                continue

            for t in range(5, n - 1):
                # Features available strictly up to step t
                lag1 = pcu[t]
                lag2 = pcu[t - 1]
                lag3 = pcu[t - 2]
                lag5 = pcu[t - 4]
                roll_mean = np.mean(pcu[max(0, t - 4) : t + 1])
                roll_std = np.std(pcu[max(0, t - 4) : t + 1]) if t >= 4 else 0.0

                # Target: actual arrivals at step t + 1
                target = pcu[t + 1]

                records.append({
                    "direction": direction,
                    "lag1": lag1,
                    "lag2": lag2,
                    "lag3": lag3,
                    "lag5": lag5,
                    "roll_mean_5": roll_mean,
                    "roll_std_5": roll_std,
                    "target": target,
                })

        df_feat = pd.DataFrame(records)
        if df_feat.empty:
            return pd.DataFrame(), pd.Series()

        # One-hot encode direction
        df_encoded = pd.get_dummies(df_feat, columns=["direction"], drop_first=True)
        X = df_encoded.drop(columns=["target"])
        y = df_encoded["target"]
        return X, y

    def train_and_evaluate(
        self,
        df_arrivals: pd.DataFrame,
        model_type: str = "gradient_boosting",
    ) -> Dict[str, Any]:
        """
        Chronologically splits data (70% train, 15% val, 15% test).
        Trains model and compares against Persistence and Moving Average baselines.
        """
        X, y = self._prepare_time_series_features(df_arrivals)
        if X.empty or len(X) < 50:
            raise ValueError("Insufficient arrival samples to train predictive model.")

        self.feature_names = list(X.columns)

        # ── Chronological Split (Strictly No Leakage) ──
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        # ── 1. Baseline: Persistence Forecast (y_pred = lag1) ──
        y_test_persistence = X_test["lag1"].values
        mae_persistence = mean_absolute_error(y_test, y_test_persistence)
        rmse_persistence = np.sqrt(mean_squared_error(y_test, y_test_persistence))

        # ── 2. Baseline: Moving Average (5-step mean) ──
        y_test_ma = X_test["roll_mean_5"].values
        mae_ma = mean_absolute_error(y_test, y_test_ma)
        rmse_ma = np.sqrt(mean_squared_error(y_test, y_test_ma))

        # ── 3. Train Machine Learning Model ──
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=60, max_depth=4, learning_rate=0.08, random_state=42)

        model.fit(X_train, y_train)
        self.model = model
        self.is_trained = True

        # Test Evaluation
        y_pred = model.predict(X_test)
        mae_ml = mean_absolute_error(y_test, y_pred)
        rmse_ml = np.sqrt(mean_squared_error(y_test, y_pred))
        r2_ml = r2_score(y_test, y_pred) if np.var(y_test) > 0 else 0.0

        improvement_vs_persistence = ((mae_persistence - mae_ml) / (mae_persistence + 1e-6)) * 100

        self.metrics = {
            "model_type": model_type,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "ml_mae": round(float(mae_ml), 4),
            "ml_rmse": round(float(rmse_ml), 4),
            "ml_r2": round(float(r2_ml), 4),
            "baseline_persistence_mae": round(float(mae_persistence), 4),
            "baseline_persistence_rmse": round(float(rmse_persistence), 4),
            "baseline_moving_avg_mae": round(float(mae_ma), 4),
            "baseline_moving_avg_rmse": round(float(rmse_ma), 4),
            "mae_improvement_percent": round(float(improvement_vs_persistence), 2),
            "feature_importance": {
                f: round(float(imp), 4)
                for f, imp in zip(X.columns, model.feature_importances_)
            } if hasattr(model, "feature_importances_") else {},
        }

        # Save model
        self.save_model()
        logger.info(f"ML Model trained: MAE={mae_ml:.4f} vs Persistence={mae_persistence:.4f}")
        return self.metrics

    @property
    def model_name(self) -> str:
        if self.metrics and "model_type" in self.metrics:
            return str(self.metrics["model_type"]).replace("_", " ").title()
        return "Gradient Boosting Regressor"

    def predict_next(
        self,
        history: List[Any],
        horizon: int = 30,
    ) -> List[float]:
        """
        Generates an autoregressive multi-step trajectory prediction for the next `horizon` seconds.
        Accepts StepObservation objects, dicts, or raw numeric sequences.
        """
        if not history:
            return [0.0] * horizon

        recent_pcu: List[float] = []
        for item in history:
            if hasattr(item, "approaches"):
                # Sum arrival PCUs across all approaches in StepObservation
                step_pcu = sum(a.arrivals_last_step.to_pcu() for a in item.approaches.values())
                recent_pcu.append(step_pcu)
            elif isinstance(item, dict) and "total_queue_pcu" in item:
                recent_pcu.append(float(item["total_queue_pcu"]))
            elif isinstance(item, (int, float)):
                recent_pcu.append(float(item))
            else:
                recent_pcu.append(0.0)

        if not recent_pcu:
            recent_pcu = [0.0]

        predictions: List[float] = []
        buffer = list(recent_pcu[-10:])
        for _ in range(horizon):
            pred = self.predict_short_term_arrivals(buffer, direction="UP")
            predictions.append(round(pred, 2))
            buffer.append(pred)
            if len(buffer) > 15:
                buffer.pop(0)

        return predictions

    def predict_short_term_arrivals(
        self,
        recent_pcu_history: List[float],
        direction: str = "UP",
    ) -> float:
        """
        Fast real-time inference for a single approach given recent arrival history.
        Falls back to rolling mean or lag-1 if ML model is unavailable.
        """
        if not recent_pcu_history:
            return 0.0

        lag1 = float(recent_pcu_history[-1])
        lag2 = float(recent_pcu_history[-2]) if len(recent_pcu_history) >= 2 else lag1
        lag3 = float(recent_pcu_history[-3]) if len(recent_pcu_history) >= 3 else lag2
        lag5 = float(recent_pcu_history[-5]) if len(recent_pcu_history) >= 5 else lag3
        roll_mean = float(np.mean(recent_pcu_history[-5:]))
        roll_std = float(np.std(recent_pcu_history[-5:])) if len(recent_pcu_history) >= 2 else 0.0

        if not self.is_trained or self.model is None or not self.feature_names:
            # Fallback to rolling average
            return round(roll_mean, 2)

        # Assemble feature vector matching trained model columns
        feat_dict = {
            "lag1": lag1,
            "lag2": lag2,
            "lag3": lag3,
            "lag5": lag5,
            "roll_mean_5": roll_mean,
            "roll_std_5": roll_std,
        }
        for f in self.feature_names:
            if f.startswith("direction_"):
                feat_dict[f] = 1 if f == f"direction_{direction}" else 0

        df_row = pd.DataFrame([feat_dict]).reindex(columns=self.feature_names, fill_value=0)
        try:
            pred = float(self.model.predict(df_row)[0])
            return max(0.0, round(pred, 2))
        except Exception:
            return round(roll_mean, 2)

    def save_model(self, path: Optional[Path] = None):
        p = path or DEFAULT_MODEL_PATH
        if self.model:
            joblib.dump({"model": self.model, "metrics": self.metrics, "features": self.feature_names}, p)

    def load_model(self, path: Optional[Path] = None) -> bool:
        p = path or DEFAULT_MODEL_PATH
        if not p.exists():
            return False
        try:
            data = joblib.load(p)
            self.model = data["model"]
            self.metrics = data.get("metrics", {})
            self.feature_names = data.get("features", [])
            self.is_trained = True
            return True
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")
            return False


# Singleton forecaster instance
forecaster = TrafficDemandForecaster()

