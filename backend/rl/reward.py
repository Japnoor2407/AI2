"""
Configurable multi-objective reward calculator for TrafficSignalEnv.

Reward components:
1. Queue improvement: rewards reductions in total intersection queue.
2. Throughput reward: rewards vehicles discharged through the green approaches.
3. Waiting time penalty: penalizes high average vehicle delay.
4. Starvation penalty: penalizes approaches that wait > 80s without green.
5. Signal switching penalty: prevents excessive phase oscillation.
6. Traffic balance penalty: prevents ignoring lower-volume approaches.
"""

from typing import Dict, Tuple
import numpy as np

from backend.core.schema import ApproachState
from backend.rl.config import RewardWeights, STARVATION_THRESHOLD_SEC


class RewardCalculator:
    """
    Computes a balanced, normalized reward and detailed component breakdown.
    """

    def __init__(self, weights: RewardWeights = None):
        self.weights = weights or RewardWeights()

    def calculate(
        self,
        prev_approaches: Dict[str, ApproachState],
        curr_approaches: Dict[str, ApproachState],
        throughput_discharged_pcu: float,
        duration_sec: int,
        phase_switched: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates normalized reward for the elapsed action interval.
        Returns:
            (total_reward, breakdown_dict)
        """
        # 1. Queue improvement: Q_before - Q_after
        q_before = sum(a.estimated_queue_pcu for a in prev_approaches.values())
        q_after = sum(a.estimated_queue_pcu for a in curr_approaches.values())
        delta_q = q_before - q_after
        # Normalize: 10 PCU reduction = +1.0
        queue_comp = float(np.clip(delta_q / 10.0, -2.0, 2.0))

        # 2. Throughput reward
        # Theoretical max discharge during green = ~1.2 PCU/s (for dual-approach green) * duration
        max_possible_throughput = max(1.0, 1.2 * duration_sec)
        throughput_comp = float(np.clip(throughput_discharged_pcu / max_possible_throughput, 0.0, 2.0))

        # 3. Waiting time penalty
        avg_wait = sum(a.average_wait_time_sec for a in curr_approaches.values()) / max(1, len(curr_approaches))
        # 60s average wait = -1.0 penalty
        wait_penalty = float(np.clip(avg_wait / 60.0, 0.0, 3.0))

        # 4. Starvation penalty: severe penalty for any approach waiting > STARVATION_THRESHOLD_SEC
        starvation_overage = 0.0
        for a in curr_approaches.values():
            if a.time_since_last_served_sec > STARVATION_THRESHOLD_SEC:
                starvation_overage += (a.time_since_last_served_sec - STARVATION_THRESHOLD_SEC)
        starvation_penalty = float(np.clip(starvation_overage / 40.0, 0.0, 3.0))

        # 5. Signal switching penalty: discourage switching every decision
        switching_penalty = 1.0 if phase_switched else 0.0

        # 6. Traffic balance penalty: penalize large variance across approach queues
        q_values = [a.estimated_queue_pcu for a in curr_approaches.values()]
        q_std = float(np.std(q_values)) if len(q_values) > 1 else 0.0
        balance_penalty = float(np.clip(q_std / 25.0, 0.0, 2.0))

        # 7. Jain's Fairness Index across waiting times: (sum(w))^2 / (n * sum(w^2))
        wait_times = [max(1.0, a.average_wait_time_sec) for a in curr_approaches.values()]
        sum_w = sum(wait_times)
        sum_sq_w = sum(w * w for w in wait_times)
        n_approaches = max(1, len(wait_times))
        jains_index = float((sum_w ** 2) / (n_approaches * sum_sq_w)) if sum_sq_w > 0 else 1.0
        fairness_comp = float(np.clip(jains_index, 0.0, 1.0))
        fairness_weight = getattr(self.weights, "fairness", 0.05)

        # Combined weighted sum
        total_reward = (
            self.weights.queue_improvement * queue_comp
            + self.weights.throughput * throughput_comp
            - self.weights.waiting_time * wait_penalty
            - self.weights.starvation * starvation_penalty
            - self.weights.switching * switching_penalty
            - self.weights.balance * balance_penalty
            + fairness_weight * (fairness_comp - 0.5)
        )

        breakdown = {
            "reward_total": round(total_reward, 4),
            "comp_queue_improvement": round(queue_comp, 4),
            "comp_throughput": round(throughput_comp, 4),
            "comp_wait_penalty": round(wait_penalty, 4),
            "comp_starvation_penalty": round(starvation_penalty, 4),
            "comp_switching_penalty": round(switching_penalty, 4),
            "comp_balance_penalty": round(balance_penalty, 4),
            "comp_fairness": round(fairness_comp, 4),
            "jains_fairness_index": round(jains_index, 4),
            "raw_delta_queue_pcu": round(delta_q, 2),
            "raw_throughput_pcu": round(throughput_discharged_pcu, 2),
            "raw_avg_wait_sec": round(avg_wait, 2),
            "raw_starvation_overage_sec": round(starvation_overage, 2),
        }

        return float(total_reward), breakdown

