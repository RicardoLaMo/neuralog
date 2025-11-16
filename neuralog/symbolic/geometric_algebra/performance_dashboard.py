"""Performance Monitoring Dashboard for NeuraLog GA System.

Provides real-time monitoring, visualization, and performance tracking
for GPU-accelerated policy verification and reasoning.

Components:
- Real-time metric collection
- Performance visualization
- Alert/threshold monitoring
- Historical tracking
- Multi-GPU aggregation
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""

    timestamp: float
    throughput_facts_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    gpu_memory_percent: float
    gpu_utilization_percent: float
    accuracy: float
    f1_score: float
    calibration_error: float
    policy_count: int
    fact_count: int


class PerformanceDashboard:
    """Real-time performance monitoring dashboard.

    Tracks:
    - Throughput (facts/sec)
    - Latency (P50, P95, P99)
    - GPU memory utilization
    - Accuracy metrics
    - Calibration quality
    - Policy performance

    Attributes:
        snapshots: Historical performance data
        alerts: Performance alert log
        update_interval: Seconds between updates
    """

    def __init__(self, update_interval: float = 1.0):
        """Initialize performance dashboard.

        Args:
            update_interval: Update frequency in seconds
        """
        self.snapshots: List[PerformanceSnapshot] = []
        self.alerts: List[Tuple[float, str]] = []
        self.update_interval = update_interval
        self.start_time = time.time()

        # Thresholds for alerts
        self.latency_threshold_ms = 50.0  # Alert if P99 > 50ms
        self.throughput_threshold = 5000.0  # Alert if < 5k facts/sec
        self.calibration_threshold = 0.1  # Alert if ECE > 0.1
        self.memory_threshold_percent = 90.0  # Alert if > 90%

    def record_snapshot(
        self,
        throughput: float,
        latencies: Dict[str, float],
        memory_percent: float,
        gpu_utilization: float,
        accuracy: float,
        f1_score: float,
        calibration_error: float,
        policy_count: int,
        fact_count: int,
    ) -> None:
        """Record performance snapshot.

        Args:
            throughput: Facts per second
            latencies: Dict with p50, p95, p99 keys
            memory_percent: GPU memory utilization %
            gpu_utilization: GPU utilization %
            accuracy: Model accuracy [0-1]
            f1_score: F1 score [0-1]
            calibration_error: ECE [0-1]
            policy_count: Number of policies
            fact_count: Number of facts processed
        """
        snapshot = PerformanceSnapshot(
            timestamp=time.time() - self.start_time,
            throughput_facts_per_sec=throughput,
            latency_p50_ms=latencies.get("p50", 0.0),
            latency_p95_ms=latencies.get("p95", 0.0),
            latency_p99_ms=latencies.get("p99", 0.0),
            gpu_memory_percent=memory_percent,
            gpu_utilization_percent=gpu_utilization,
            accuracy=accuracy,
            f1_score=f1_score,
            calibration_error=calibration_error,
            policy_count=policy_count,
            fact_count=fact_count,
        )

        self.snapshots.append(snapshot)
        self._check_alerts(snapshot)

    def _check_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Check for performance alerts.

        Args:
            snapshot: Performance snapshot
        """
        if snapshot.latency_p99_ms > self.latency_threshold_ms:
            self.alerts.append(
                (
                    snapshot.timestamp,
                    f"High P99 latency: {snapshot.latency_p99_ms:.1f}ms",
                )
            )

        if snapshot.throughput_facts_per_sec < self.throughput_threshold:
            self.alerts.append(
                (
                    snapshot.timestamp,
                    f"Low throughput: {snapshot.throughput_facts_per_sec:.0f} facts/sec",
                )
            )

        if snapshot.calibration_error > self.calibration_threshold:
            self.alerts.append(
                (
                    snapshot.timestamp,
                    f"Poor calibration: ECE={snapshot.calibration_error:.3f}",
                )
            )

        if snapshot.gpu_memory_percent > self.memory_threshold_percent:
            self.alerts.append(
                (
                    snapshot.timestamp,
                    f"High GPU memory: {snapshot.gpu_memory_percent:.1f}%",
                )
            )

    def get_summary(self) -> str:
        """Get performance summary.

        Returns:
            Formatted summary string
        """
        if not self.snapshots:
            return "No snapshots recorded"

        latest = self.snapshots[-1]

        lines = [
            "=" * 70,
            "Performance Dashboard Summary",
            "=" * 70,
            f"Uptime: {latest.timestamp:.1f}s",
            "",
            "Throughput & Latency:",
            f"  Throughput: {latest.throughput_facts_per_sec:.0f} facts/sec",
            f"  Latency P50: {latest.latency_p50_ms:.2f}ms",
            f"  Latency P95: {latest.latency_p95_ms:.2f}ms",
            f"  Latency P99: {latest.latency_p99_ms:.2f}ms",
            "",
            "GPU & Memory:",
            f"  GPU Utilization: {latest.gpu_utilization_percent:.1f}%",
            f"  Memory Usage: {latest.gpu_memory_percent:.1f}%",
            "",
            "Quality Metrics:",
            f"  Accuracy: {latest.accuracy:.3f}",
            f"  F1 Score: {latest.f1_score:.3f}",
            f"  Calibration (ECE): {latest.calibration_error:.4f}",
            "",
            "System:",
            f"  Policies: {latest.policy_count}",
            f"  Facts Processed: {latest.fact_count}",
        ]

        if self.alerts:
            lines.extend(
                [
                    "",
                    f"Recent Alerts ({len(self.alerts)}):",
                ]
            )
            for timestamp, alert in self.alerts[-5:]:
                lines.append(f"  [{timestamp:.1f}s] {alert}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def plot_performance(self, output_file: Optional[str] = None) -> None:
        """Plot performance metrics.

        Args:
            output_file: Optional file to save plot
        """
        if not VISUALIZATION_AVAILABLE or not self.snapshots:
            print("Visualization not available or no snapshots")
            return

        snapshots = self.snapshots
        timestamps = [s.timestamp for s in snapshots]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("NeuraLog GA System Performance Dashboard", fontsize=16)

        # Throughput
        ax = axes[0, 0]
        throughputs = [s.throughput_facts_per_sec for s in snapshots]
        ax.plot(timestamps, throughputs, "b-", linewidth=2)
        ax.axhline(self.throughput_threshold, color="r", linestyle="--", label="Threshold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Throughput (facts/sec)")
        ax.set_title("Throughput")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Latency
        ax = axes[0, 1]
        p50s = [s.latency_p50_ms for s in snapshots]
        p95s = [s.latency_p95_ms for s in snapshots]
        p99s = [s.latency_p99_ms for s in snapshots]
        ax.plot(timestamps, p50s, "g-", label="P50", linewidth=2)
        ax.plot(timestamps, p95s, "y-", label="P95", linewidth=2)
        ax.plot(timestamps, p99s, "r-", label="P99", linewidth=2)
        ax.axhline(self.latency_threshold_ms, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # GPU Utilization & Memory
        ax = axes[1, 0]
        gpu_utils = [s.gpu_utilization_percent for s in snapshots]
        gpu_mems = [s.gpu_memory_percent for s in snapshots]
        ax.plot(timestamps, gpu_utils, "b-", label="GPU Util", linewidth=2)
        ax.plot(timestamps, gpu_mems, "m-", label="GPU Memory", linewidth=2)
        ax.axhline(90, color="r", linestyle="--", alpha=0.5, label="Alert")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Utilization (%)")
        ax.set_title("GPU Utilization")
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Quality Metrics
        ax = axes[1, 1]
        accs = [s.accuracy for s in snapshots]
        f1s = [s.f1_score for s in snapshots]
        cals = [s.calibration_error for s in snapshots]
        ax.plot(timestamps, accs, "g-", label="Accuracy", linewidth=2)
        ax.plot(timestamps, f1s, "b-", label="F1 Score", linewidth=2)
        ax.plot(timestamps, cals, "r-", label="ECE (calib)", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Score")
        ax.set_title("Quality Metrics")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150)
            print(f"Performance plot saved to {output_file}")
        else:
            plt.show()

    def export_metrics(self) -> Dict:
        """Export metrics as dictionary.

        Returns:
            Dictionary with all metrics
        """
        if not self.snapshots:
            return {}

        latest = self.snapshots[-1]

        # Compute averages
        avg_throughput = sum(s.throughput_facts_per_sec for s in self.snapshots) / len(
            self.snapshots
        )
        avg_p99 = sum(s.latency_p99_ms for s in self.snapshots) / len(self.snapshots)
        avg_accuracy = sum(s.accuracy for s in self.snapshots) / len(self.snapshots)

        return {
            "latest_throughput": latest.throughput_facts_per_sec,
            "avg_throughput": avg_throughput,
            "latest_p99_latency_ms": latest.latency_p99_ms,
            "avg_p99_latency_ms": avg_p99,
            "gpu_utilization_percent": latest.gpu_utilization_percent,
            "gpu_memory_percent": latest.gpu_memory_percent,
            "latest_accuracy": latest.accuracy,
            "avg_accuracy": avg_accuracy,
            "latest_f1": latest.f1_score,
            "latest_calibration_error": latest.calibration_error,
            "snapshot_count": len(self.snapshots),
            "alert_count": len(self.alerts),
        }


class GPUMetricsCollector:
    """Collects GPU-specific metrics.

    Tracks GPU memory, utilization, compute capacity.
    """

    def __init__(self, device: int = 0):
        """Initialize GPU metrics collector.

        Args:
            device: GPU device ID
        """
        self.device = device
        self.device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics.

        Returns:
            Dict with allocated, reserved, free memory in GB
        """
        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}

        with torch.cuda.device(self.device):
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            free = total - allocated

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "free_gb": free,
                "total_gb": total,
                "utilization_percent": (allocated / total) * 100,
            }

    def get_utilization(self) -> float:
        """Get estimated GPU utilization percentage.

        Returns:
            Utilization as percentage [0-100]
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except (ImportError, Exception):
            # Fallback: estimate from memory usage
            stats = self.get_memory_stats()
            return stats.get("utilization_percent", 0.0)
