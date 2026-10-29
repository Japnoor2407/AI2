"""
Evaluation of Vehicle Detection & Tracking against Ground Truth Annotations.
"""

from typing import Dict, Any


class DetectionEvaluator:
    """
    Evaluates automated vision tripwire counts against manual ground-truth counts.
    Computes Precision, Recall, F1-Score, and Mean Absolute Percentage Error (MAPE).
    """

    @staticmethod
    def evaluate(predicted_counts: Dict[str, int], ground_truth: Dict[str, int]) -> Dict[str, Any]:
        classes = set(predicted_counts.keys()).union(set(ground_truth.keys()))
        per_class_metrics = {}

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for c in classes:
            pred = predicted_counts.get(c, 0)
            gt = ground_truth.get(c, 0)

            tp = min(pred, gt)
            fp = max(0, pred - gt)
            fn = max(0, gt - pred)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-6, precision + recall)
            abs_err = abs(pred - gt)
            mape = (abs_err / max(1, gt)) * 100.0

            per_class_metrics[c] = {
                "ground_truth": gt,
                "detected": pred,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                "mape_percent": round(mape, 2),
            }

        overall_precision = total_tp / max(1, total_tp + total_fp)
        overall_recall = total_tp / max(1, total_tp + total_fn)
        overall_f1 = 2 * overall_precision * overall_recall / max(1e-6, overall_precision + overall_recall)

        return {
            "overall_precision": round(overall_precision, 3),
            "overall_recall": round(overall_recall, 3),
            "overall_f1_score": round(overall_f1, 3),
            "total_detected": sum(predicted_counts.values()),
            "total_ground_truth": sum(ground_truth.values()),
            "per_class": per_class_metrics,
        }

