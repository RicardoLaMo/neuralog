"""Verification metrics and evaluation framework.

Tracks performance of policy verification, LLM integration, and reasoning quality.
Computes calibration, accuracy, latency, and throughput metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ConfusionMatrices:
    """Per-policy confusion matrices."""

    tp: int = 0  # True positives
    tn: int = 0  # True negatives
    fp: int = 0  # False positives
    fn: int = 0  # False negatives
    policy_name: str = "default"

    def accuracy(self) -> float:
        """Compute accuracy."""
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def precision(self) -> float:
        """Compute precision (positive predictive value)."""
        total = self.tp + self.fp
        return self.tp / total if total > 0 else 0.0

    def recall(self) -> float:
        """Compute recall (sensitivity/true positive rate)."""
        total = self.tp + self.fn
        return self.tp / total if total > 0 else 0.0

    def f1_score(self) -> float:
        """Compute F1 score."""
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    def specificity(self) -> float:
        """Compute specificity (true negative rate)."""
        total = self.tn + self.fp
        return self.tn / total if total > 0 else 0.0

    def false_positive_rate(self) -> float:
        """Compute false positive rate."""
        total = self.fp + self.tn
        return self.fp / total if total > 0 else 0.0

    def false_negative_rate(self) -> float:
        """Compute false negative rate."""
        total = self.fn + self.tp
        return self.fn / total if total > 0 else 0.0


@dataclass
class CalibrationMetrics:
    """Confidence calibration quality metrics."""

    expected_calibration_error: float = 0.0  # ECE
    maximum_calibration_error: float = 0.0  # MCE
    brier_score: float = 0.0
    log_loss: float = 0.0
    auc_roc: float = 0.0
    predictions: List[float] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)

    def is_well_calibrated(self, threshold: float = 0.1) -> bool:
        """Check if predictions are well-calibrated."""
        return self.expected_calibration_error < threshold


@dataclass
class LatencyMetrics:
    """Performance latency metrics (in milliseconds)."""

    p50: float = 0.0  # Median
    p95: float = 0.0  # 95th percentile
    p99: float = 0.0  # 99th percentile
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    latencies: List[float] = field(default_factory=list)

    def update(self, latency_ms: float):
        """Add latency measurement."""
        self.latencies.append(latency_ms)

    def compute(self):
        """Compute percentile statistics."""
        if not self.latencies:
            return

        arr = np.array(self.latencies)
        self.p50 = float(np.percentile(arr, 50))
        self.p95 = float(np.percentile(arr, 95))
        self.p99 = float(np.percentile(arr, 99))
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr))
        self.min = float(np.min(arr))
        self.max = float(np.max(arr))


@dataclass
class ThroughputMetrics:
    """Throughput and resource utilization metrics."""

    facts_per_second: float = 0.0
    policies_per_second: float = 0.0
    gpu_memory_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    batch_size: int = 0
    total_facts_processed: int = 0
    total_time_seconds: float = 0.0


@dataclass
class VerificationMetrics:
    """Complete verification metrics."""

    # Accuracy metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0

    # Loss metrics
    mean_implication_loss: float = 0.0
    mean_fp_loss: float = 0.0
    mean_fn_loss: float = 0.0
    mean_total_loss: float = 0.0

    # Truth degree distribution
    mean_truth_degree: float = 0.0
    std_truth_degree: float = 0.0
    satisfaction_rate: float = 0.0  # % satisfying policies

    # Calibration
    calibration: CalibrationMetrics = field(default_factory=CalibrationMetrics)

    # Performance
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)

    # Per-policy metrics
    confusion_matrices: Dict[str, ConfusionMatrices] = field(
        default_factory=dict
    )

    # Metadata
    num_facts: int = 0
    num_policies: int = 0
    device: str = "cpu"
    timestamp: str = ""


class MetricsCollector:
    """Collects and aggregates verification metrics.

    Supports:
    - Streaming metric updates (online computation)
    - Batch metric computation
    - GPU memory tracking
    - Calibration curve computation
    """

    def __init__(self, device: str = "cpu"):
        """Initialize metrics collector.

        Args:
            device: PyTorch device for tracking
        """
        self.device = device
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.truth_degrees = []
        self.predictions = []
        self.labels = []
        self.latencies_ms = []
        self.fp_losses = []
        self.fn_losses = []
        self.impl_losses = []
        self.confusion_matrices: Dict[str, ConfusionMatrices] = {}
        self.total_facts = 0
        self.total_policies = 0

    def update_truth_degree(self, truth_degree: float):
        """Add truth degree measurement."""
        self.truth_degrees.append(truth_degree)

    def update_loss(
        self,
        impl_loss: float,
        fp_loss: float,
        fn_loss: float,
    ):
        """Add loss measurements."""
        self.impl_losses.append(impl_loss)
        self.fp_losses.append(fp_loss)
        self.fn_losses.append(fn_loss)

    def update_prediction(
        self,
        prediction: float,
        label: int,
        policy_name: str = "default",
    ):
        """Add prediction-label pair for metrics.

        Args:
            prediction: Predicted truth degree [0, 1]
            label: Ground truth label (0 or 1)
            policy_name: Associated policy name
        """
        self.predictions.append(prediction)
        self.labels.append(label)

        # Update confusion matrix
        if policy_name not in self.confusion_matrices:
            self.confusion_matrices[policy_name] = ConfusionMatrices(
                policy_name=policy_name
            )

        cm = self.confusion_matrices[policy_name]
        predicted_label = 1 if prediction > 0.5 else 0

        if predicted_label == 1 and label == 1:
            cm.tp += 1
        elif predicted_label == 1 and label == 0:
            cm.fp += 1
        elif predicted_label == 0 and label == 1:
            cm.fn += 1
        else:
            cm.tn += 1

    def update_latency(self, latency_ms: float):
        """Add latency measurement."""
        self.latencies_ms.append(latency_ms)

    def compute_metrics(
        self,
        num_policies: int = 1,
    ) -> VerificationMetrics:
        """Compute aggregated metrics.

        Args:
            num_policies: Number of policies evaluated

        Returns:
            VerificationMetrics object
        """
        from datetime import datetime

        metrics = VerificationMetrics()
        metrics.num_facts = self.total_facts
        metrics.num_policies = num_policies
        metrics.device = self.device
        metrics.timestamp = datetime.now().isoformat()

        # Truth degree metrics
        if self.truth_degrees:
            truth_arr = np.array(self.truth_degrees)
            metrics.mean_truth_degree = float(np.mean(truth_arr))
            metrics.std_truth_degree = float(np.std(truth_arr))
            metrics.satisfaction_rate = float(
                (truth_arr > 0.5).mean()
            )

        # Loss metrics
        if self.impl_losses:
            metrics.mean_implication_loss = float(np.mean(self.impl_losses))
        if self.fp_losses:
            metrics.mean_fp_loss = float(np.mean(self.fp_losses))
        if self.fn_losses:
            metrics.mean_fn_loss = float(np.mean(self.fn_losses))

        mean_total = float(
            np.mean(self.impl_losses) +
            np.mean(self.fp_losses) +
            np.mean(self.fn_losses)
        ) if (self.impl_losses and self.fp_losses and self.fn_losses) else 0.0
        metrics.mean_total_loss = mean_total

        # Per-policy metrics
        metrics.confusion_matrices = self.confusion_matrices

        # Aggregate accuracy metrics across all policies
        if self.confusion_matrices:
            total_tp = sum(cm.tp for cm in self.confusion_matrices.values())
            total_tn = sum(cm.tn for cm in self.confusion_matrices.values())
            total_fp = sum(cm.fp for cm in self.confusion_matrices.values())
            total_fn = sum(cm.fn for cm in self.confusion_matrices.values())

            total = total_tp + total_tn + total_fp + total_fn
            if total > 0:
                metrics.accuracy = (total_tp + total_tn) / total
                metrics.precision = (
                    total_tp / (total_tp + total_fp)
                    if (total_tp + total_fp) > 0
                    else 0.0
                )
                metrics.recall = (
                    total_tp / (total_tp + total_fn)
                    if (total_tp + total_fn) > 0
                    else 0.0
                )
                metrics.f1 = (
                    2 * (metrics.precision * metrics.recall) /
                    (metrics.precision + metrics.recall)
                    if (metrics.precision + metrics.recall) > 0
                    else 0.0
                )
                metrics.specificity = (
                    total_tn / (total_tn + total_fp)
                    if (total_tn + total_fp) > 0
                    else 0.0
                )

        # Calibration metrics
        if self.predictions and self.labels:
            metrics.calibration = self._compute_calibration()

        # Latency metrics
        if self.latencies_ms:
            metrics.latency.latencies = self.latencies_ms
            metrics.latency.compute()

        # Throughput metrics
        if self.latencies_ms:
            total_time_s = sum(self.latencies_ms) / 1000
            metrics.throughput.facts_per_second = (
                self.total_facts / total_time_s
                if total_time_s > 0
                else 0.0
            )
            metrics.throughput.total_facts_processed = self.total_facts
            metrics.throughput.total_time_seconds = total_time_s

        return metrics

    def _compute_calibration(self) -> CalibrationMetrics:
        """Compute calibration metrics."""
        cal = CalibrationMetrics()
        cal.predictions = self.predictions
        cal.labels = self.labels

        if not self.predictions or not self.labels:
            return cal

        preds = np.array(self.predictions)
        labels = np.array(self.labels)

        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, 11)
        bin_indices = np.digitize(preds, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, 9)

        ece = 0.0
        for bin_idx in range(10):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = preds[mask].mean()
                bin_weight = mask.sum() / len(preds)
                ece += bin_weight * abs(bin_acc - bin_conf)

        cal.expected_calibration_error = ece

        # Maximum Calibration Error
        mce = 0.0
        for bin_idx in range(10):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = preds[mask].mean()
                mce = max(mce, abs(bin_acc - bin_conf))

        cal.maximum_calibration_error = mce

        # Brier score
        cal.brier_score = float(((preds - labels) ** 2).mean())

        # Log loss
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        cal.log_loss = float(
            -(labels * np.log(preds_clipped) +
              (1 - labels) * np.log(1 - preds_clipped)).mean()
        )

        # AUC-ROC (requires sklearn)
        try:
            from sklearn.metrics import roc_auc_score
            cal.auc_roc = roc_auc_score(labels, preds)
        except ImportError:
            cal.auc_roc = 0.0

        return cal

    def log_summary(self) -> str:
        """Generate summary log of metrics."""
        metrics = self.compute_metrics()

        lines = [
            "=" * 60,
            "Verification Metrics Summary",
            "=" * 60,
            f"Device: {metrics.device}",
            f"Facts Processed: {metrics.num_facts}",
            f"Policies: {metrics.num_policies}",
            "",
            "Accuracy Metrics:",
            f"  Accuracy:  {metrics.accuracy:.4f}",
            f"  Precision: {metrics.precision:.4f}",
            f"  Recall:    {metrics.recall:.4f}",
            f"  F1 Score:  {metrics.f1:.4f}",
            f"  Specificity: {metrics.specificity:.4f}",
            "",
            "Loss Metrics:",
            f"  Implication Loss: {metrics.mean_implication_loss:.6f}",
            f"  FP Loss: {metrics.mean_fp_loss:.6f}",
            f"  FN Loss: {metrics.mean_fn_loss:.6f}",
            "",
            "Truth Degree Metrics:",
            f"  Mean: {metrics.mean_truth_degree:.4f}",
            f"  Std:  {metrics.std_truth_degree:.4f}",
            f"  Satisfaction Rate: {metrics.satisfaction_rate:.4f}",
            "",
            "Calibration:",
            f"  ECE: {metrics.calibration.expected_calibration_error:.4f}",
            f"  MCE: {metrics.calibration.maximum_calibration_error:.4f}",
            f"  Brier Score: {metrics.calibration.brier_score:.6f}",
            "",
            "Latency (ms):",
            f"  P50: {metrics.latency.p50:.2f}",
            f"  P95: {metrics.latency.p95:.2f}",
            f"  P99: {metrics.latency.p99:.2f}",
            f"  Mean: {metrics.latency.mean:.2f}",
            "",
            "Throughput:",
            f"  Facts/sec: {metrics.throughput.facts_per_second:.0f}",
            f"  Total Time: {metrics.throughput.total_time_seconds:.2f}s",
            "=" * 60,
        ]

        return "\n".join(lines)
