import logging
from typing import Any, Dict, List, Optional, Tuple
from backend.core.schema import (
    ApproachState, ControllerDecision, ControllerMode, PhaseConfig, SignalState,
)
from backend.core.config import (
    MIN_GREEN_SEC, MAX_GREEN_SEC, YELLOW_CLEARANCE_SEC, ALL_RED_CLEARANCE_SEC,
    STARVATION_THRESHOLD_SEC, DEFAULT_ADAPTIVE_WEIGHTS,
)

logger = logging.getLogger(__name__)


class AdaptiveController:
    """
    Demand-responsive adaptive signal controller.
    
    Demand score per phase:
      score = wq*Q_norm + wa*A_norm + ww*W_norm + wt*T_norm
    where:
      Q_norm = normalized estimated queue (PCU)
      A_norm = normalized arrival rate (PCU/min)
      W_norm = normalized accumulated wait time (sec)
      T_norm = normalized time since last served (sec)  [anti-starvation]
    
    Safety invariants enforced:
    - Conflicting approaches never get green simultaneously
    - Yellow and all-red clearance intervals always respected
    - No approach waits > STARVATION_THRESHOLD_SEC (priority override)
    - Green duration clamped to [MIN_GREEN, MAX_GREEN]
    """

    def __init__(
        self,
        phases: List[PhaseConfig],
        weights: Optional[Dict[str, float]] = None,
        yellow_sec: int = YELLOW_CLEARANCE_SEC,
        all_red_sec: int = ALL_RED_CLEARANCE_SEC,
        min_green: int = MIN_GREEN_SEC,
        max_green: int = MAX_GREEN_SEC,
        starvation_threshold: int = STARVATION_THRESHOLD_SEC,
        mode: ControllerMode = ControllerMode.ADAPTIVE,
        forecaster: Optional[Any] = None,
    ):
        self.phases = phases
        self.weights = weights or DEFAULT_ADAPTIVE_WEIGHTS
        self.yellow_sec = yellow_sec
        self.all_red_sec = all_red_sec
        self.min_green = min_green
        self.max_green = max_green
        self.starvation_threshold = starvation_threshold
        self.mode = mode
        self.forecaster = forecaster

        self.current_phase_index: int = 0
        self.current_signal_state: SignalState = SignalState.GREEN
        self.elapsed_in_state: int = 0
        self.allocated_green: int = min_green
        self.target_phase_index: Optional[int] = None
        self.demand_scores: Dict[str, float] = {}
        self.predicted_arrivals: Dict[str, float] = {}
        self._approach_arrival_hist: Dict[str, List[float]] = {}

    def reset(self):
        self.current_phase_index = 0
        self.current_signal_state = SignalState.GREEN
        self.elapsed_in_state = 0
        self.allocated_green = self.min_green
        self.target_phase_index = None
        self.demand_scores = {}
        self.predicted_arrivals = {}
        self._approach_arrival_hist = {}

    @property
    def current_phase(self) -> PhaseConfig:
        return self.phases[self.current_phase_index]

    def _compute_demand_scores(
        self, approaches: Dict[str, ApproachState]
    ) -> Dict[str, float]:
        """Compute per-phase demand score using weighted normalized features."""
        phase_scores: Dict[str, float] = {}
        if not approaches:
            return {str(p.phase_id): 0.0 for p in self.phases}

        # Track arrival history for prediction
        for app_id, app in approaches.items():
            if app_id not in self._approach_arrival_hist:
                self._approach_arrival_hist[app_id] = []
            self._approach_arrival_hist[app_id].append(app.arrivals_last_step.to_pcu())
            if len(self._approach_arrival_hist[app_id]) > 30:
                self._approach_arrival_hist[app_id].pop(0)

        # Predict short-term arrivals if in prediction-assisted mode
        pred_map: Dict[str, float] = {}
        if self.mode == ControllerMode.PREDICTION_ASSISTED and self.forecaster is not None:
            for app_id, app in approaches.items():
                try:
                    hist = self._approach_arrival_hist.get(app_id, [])
                    pred = self.forecaster.predict_short_term_arrivals(hist, direction=app.direction)
                    pred_map[app_id] = pred
                except Exception as e:
                    logger.debug(f"Prediction fallback for {app_id}: {e}")
                    pred_map[app_id] = app.arrivals_last_step.to_pcu()
        self.predicted_arrivals = pred_map

        # Collect raw values across all approaches for normalization
        all_queues = [a.estimated_queue_pcu for a in approaches.values()]
        all_rates = [a.arrival_rate_pcu_per_min for a in approaches.values()]
        all_waits = [a.average_wait_time_sec for a in approaches.values()]
        all_times = [a.time_since_last_served_sec for a in approaches.values()]
        all_preds = list(pred_map.values()) if pred_map else [0.0]

        max_q = max(all_queues) if max(all_queues) > 0 else 1.0
        max_r = max(all_rates) if max(all_rates) > 0 else 1.0
        max_w = max(all_waits) if max(all_waits) > 0 else 1.0
        max_t = max(all_times) if max(all_times) > 0 else 1.0
        max_p = max(all_preds) if max(all_preds) > 0 else 1.0

        if self.mode == ControllerMode.PREDICTION_ASSISTED and pred_map:
            wq = 0.35
            wa = 0.20
            ww = 0.15
            wt = 0.10
            wp = 0.20
        else:
            wq = self.weights.get("wq", 0.40)
            wa = self.weights.get("wa", 0.25)
            ww = self.weights.get("ww", 0.20)
            wt = self.weights.get("wt", 0.15)
            wp = 0.0

        for phase in self.phases:
            score = 0.0
            for app_id in phase.served_approaches:
                if app_id not in approaches:
                    continue
                app = approaches[app_id]
                q_norm = app.estimated_queue_pcu / max_q
                a_norm = app.arrival_rate_pcu_per_min / max_r
                w_norm = app.average_wait_time_sec / max_w
                t_norm = app.time_since_last_served_sec / max_t
                p_norm = pred_map.get(app_id, 0.0) / max_p if wp > 0 else 0.0
                score += wq * q_norm + wa * a_norm + ww * w_norm + wt * t_norm + wp * p_norm
            phase_scores[str(phase.phase_id)] = round(score, 4)

        return phase_scores

    def _allocate_green_duration(
        self, phase: PhaseConfig, approaches: Dict[str, ApproachState]
    ) -> int:
        """Scale green time proportionally to demand, clamped to [min, max]."""
        total_demand = sum(
            approaches[a].estimated_queue_pcu
            + approaches[a].arrival_rate_pcu_per_min / 60.0
            + self.predicted_arrivals.get(a, 0.0)
            for a in phase.served_approaches
            if a in approaches
        )
        # Base: 10s + 2.5s per PCU (loosely based on saturation flow)
        raw_green = self.min_green + int(total_demand * 2.5)
        return max(self.min_green, min(self.max_green, raw_green))

    def _check_starvation(self, approaches: Dict[str, ApproachState]) -> Optional[int]:
        """Return phase_id that must get green immediately due to starvation."""
        for phase in self.phases:
            for app_id in phase.served_approaches:
                if app_id in approaches:
                    app = approaches[app_id]
                    if app.time_since_last_served_sec >= self.starvation_threshold:
                        logger.warning(f"STARVATION override: {app_id} waited {app.time_since_last_served_sec:.0f}s")
                        return phase.phase_id
        return None

    def step(
        self,
        current_step: int,
        timestamp_str: str,
        approaches: Dict[str, ApproachState],
    ) -> ControllerDecision:
        self.elapsed_in_state += 1
        active_phase = self.phases[self.current_phase_index]
        is_transitioning = self.current_signal_state in (SignalState.YELLOW, SignalState.ALL_RED)
        rationale = ""

        if self.current_signal_state == SignalState.GREEN:
            scores = self._compute_demand_scores(approaches)
            self.demand_scores = scores
            remaining = max(0, self.allocated_green - self.elapsed_in_state)

            # Starvation check for early switch
            starving_phase_id = self._check_starvation(approaches)
            force_switch = (
                starving_phase_id is not None
                and starving_phase_id != active_phase.phase_id
                and self.elapsed_in_state >= self.min_green
            )

            if force_switch:
                # Find starving phase index
                tgt_idx = next(
                    (i for i, p in enumerate(self.phases) if p.phase_id == starving_phase_id),
                    (self.current_phase_index + 1) % len(self.phases),
                )
                self.target_phase_index = tgt_idx
                self.current_signal_state = SignalState.YELLOW
                self.elapsed_in_state = 0
                rationale = (
                    f"STARVATION OVERRIDE: Phase {starving_phase_id} approach exceeded "
                    f"{self.starvation_threshold}s wait. Initiating Yellow clearance."
                )
            elif self.elapsed_in_state >= self.allocated_green:
                # Normal adaptive phase switch: choose highest-scoring next phase
                next_idx = (self.current_phase_index + 1) % len(self.phases)
                best_score = -1.0
                for i, p in enumerate(self.phases):
                    if i == self.current_phase_index:
                        continue
                    s = scores.get(str(p.phase_id), 0.0)
                    if s > best_score:
                        best_score = s
                        next_idx = i
                self.target_phase_index = next_idx
                self.current_signal_state = SignalState.YELLOW
                self.elapsed_in_state = 0
                rationale = (
                    f"Adaptive green ({self.allocated_green}s) complete. "
                    f"Best next phase: {self.phases[next_idx].name} (score={best_score:.3f}). "
                    f"Initiating Yellow clearance."
                )
            else:
                rationale = (
                    f"Holding green for {active_phase.name}. "
                    f"Phase scores: {scores}. {remaining}s remaining."
                )

        elif self.current_signal_state == SignalState.YELLOW:
            remaining = max(0, self.yellow_sec - self.elapsed_in_state)
            if self.elapsed_in_state >= self.yellow_sec:
                self.current_signal_state = SignalState.ALL_RED
                self.elapsed_in_state = 0
                rationale = "Yellow clearance complete. Entering All-Red safety interval."
            else:
                rationale = f"Yellow clearance: {remaining}s remaining."

        elif self.current_signal_state == SignalState.ALL_RED:
            remaining = max(0, self.all_red_sec - self.elapsed_in_state)
            if self.elapsed_in_state >= self.all_red_sec:
                self.current_phase_index = self.target_phase_index
                self.current_signal_state = SignalState.GREEN
                self.elapsed_in_state = 0
                active_phase = self.phases[self.current_phase_index]
                self.allocated_green = self._allocate_green_duration(active_phase, approaches)
                self.target_phase_index = None
                rationale = (
                    f"All-Red complete. Activating {active_phase.name}. "
                    f"Allocated green: {self.allocated_green}s (demand-based)."
                )
            else:
                rationale = f"All-Red junction clearance: {remaining}s remaining."

        tot_pred = round(sum(self.predicted_arrivals.values()), 2) if self.predicted_arrivals else None
        return ControllerDecision(
            step=current_step,
            timestamp_str=timestamp_str,
            mode=self.mode,
            active_phase_id=active_phase.phase_id,
            phase_name=active_phase.name,
            signal_state=self.current_signal_state,
            time_remaining_in_state=max(0, (
                self.allocated_green if self.current_signal_state == SignalState.GREEN
                else self.yellow_sec if self.current_signal_state == SignalState.YELLOW
                else self.all_red_sec
            ) - self.elapsed_in_state),
            elapsed_in_phase=self.elapsed_in_state,
            phase_demand_scores=self.demand_scores,
            predicted_arrivals_pcu=tot_pred,
            decision_rationale=rationale,
            is_transitioning=is_transitioning,
            transition_stage=self.current_signal_state.value if is_transitioning else None,
        )

