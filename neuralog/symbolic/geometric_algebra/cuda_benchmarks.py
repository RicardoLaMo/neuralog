"""Benchmarking Suite for CUDA Optimizations.

Compares PyTorch vs CUDA implementations for:
- Threshold evaluation (HARD/SOFT/MARGIN)
- Truth degree computation
- Batch policy verification
- End-to-end pipeline throughput

Targets:
- H200 GPU (Hopper architecture)
- 10k+ facts/sec performance goal
- 30-50% speedup vs PyTorch
"""

import time
from typing import Dict, List, Tuple

import torch

from neuralog.symbolic.geometric_algebra.batched_verifier import (
    BatchedPolicyVerifier,
)
from neuralog.symbolic.geometric_algebra.cuda_batched_verifier import (
    CUDABatchedPolicyVerifier,
)


class BenchmarkResult:
    """Result of a single benchmark run."""

    def __init__(
        self,
        name: str,
        pytorch_time_ms: float,
        cuda_time_ms: float,
        batch_size: int,
        device: str,
    ):
        """Initialize benchmark result.

        Args:
            name: Benchmark name
            pytorch_time_ms: PyTorch execution time
            cuda_time_ms: CUDA execution time
            batch_size: Batch size
            device: GPU device used
        """
        self.name = name
        self.pytorch_time_ms = pytorch_time_ms
        self.cuda_time_ms = cuda_time_ms
        self.batch_size = batch_size
        self.device = device

    @property
    def speedup(self) -> float:
        """Compute speedup ratio.

        Returns:
            CUDA speedup (pytorch_time / cuda_time)
        """
        if self.cuda_time_ms == 0:
            return 0.0
        return self.pytorch_time_ms / self.cuda_time_ms

    @property
    def pytorch_throughput(self) -> float:
        """PyTorch throughput in facts/sec.

        Returns:
            Facts per second
        """
        if self.pytorch_time_ms == 0:
            return 0.0
        return (self.batch_size / self.pytorch_time_ms) * 1000

    @property
    def cuda_throughput(self) -> float:
        """CUDA throughput in facts/sec.

        Returns:
            Facts per second
        """
        if self.cuda_time_ms == 0:
            return 0.0
        return (self.batch_size / self.cuda_time_ms) * 1000

    def __str__(self) -> str:
        """String representation."""
        return (
            f"{self.name:40s} | "
            f"PyTorch: {self.pytorch_time_ms:8.2f}ms | "
            f"CUDA: {self.cuda_time_ms:8.2f}ms | "
            f"Speedup: {self.speedup:5.2f}x | "
            f"CUDA: {self.cuda_throughput:7.0f} facts/sec"
        )


class CUDABenchmarkSuite:
    """Comprehensive CUDA optimization benchmark suite.

    Benchmarks:
    1. Threshold evaluation (HARD/SOFT/MARGIN)
    2. Truth degree computation
    3. Batch policy verification
    4. End-to-end pipeline
    5. Scaling behavior (batch size variation)
    """

    def __init__(self, device: str = "cuda"):
        """Initialize benchmark suite.

        Args:
            device: GPU device ("cuda" or "cpu")
        """
        self.device = device
        self.results: List[BenchmarkResult] = []
        torch.cuda.empty_cache()

    def benchmark_hard_threshold(
        self, batch_sizes: List[int] = None
    ) -> List[BenchmarkResult]:
        """Benchmark hard threshold evaluation.

        Args:
            batch_sizes: Batch sizes to test

        Returns:
            List of benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [32, 64, 128, 256, 512, 1024]

        results = []

        for batch_size in batch_sizes:
            values = torch.randn(batch_size, device=self.device)
            thresholds = torch.randn(batch_size, device=self.device)

            # PyTorch baseline
            pytorch_verifier = BatchedPolicyVerifier(device=self.device)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = pytorch_verifier._fused_threshold_eval(
                    values, thresholds, "hard"
                )
            torch.cuda.synchronize()
            pytorch_time_ms = (time.time() - start) * 10  # 100 iterations

            # CUDA optimized
            cuda_verifier = CUDABatchedPolicyVerifier(device=self.device)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = cuda_verifier._fused_threshold_eval_cuda(
                    values, thresholds, "hard"
                )
            torch.cuda.synchronize()
            cuda_time_ms = (time.time() - start) * 10

            result = BenchmarkResult(
                f"Hard Threshold (batch={batch_size})",
                pytorch_time_ms,
                cuda_time_ms,
                batch_size,
                self.device,
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_soft_threshold(
        self, batch_sizes: List[int] = None
    ) -> List[BenchmarkResult]:
        """Benchmark soft (sigmoid) threshold evaluation.

        Args:
            batch_sizes: Batch sizes to test

        Returns:
            List of benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [32, 64, 128, 256, 512, 1024]

        results = []

        for batch_size in batch_sizes:
            values = torch.randn(batch_size, device=self.device)
            thresholds = torch.randn(batch_size, device=self.device)

            # PyTorch baseline
            pytorch_verifier = BatchedPolicyVerifier(device=self.device)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = pytorch_verifier._fused_threshold_eval(
                    values, thresholds, "soft"
                )
            torch.cuda.synchronize()
            pytorch_time_ms = (time.time() - start) * 10

            # CUDA optimized
            cuda_verifier = CUDABatchedPolicyVerifier(device=self.device)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = cuda_verifier._fused_threshold_eval_cuda(
                    values, thresholds, "soft"
                )
            torch.cuda.synchronize()
            cuda_time_ms = (time.time() - start) * 10

            result = BenchmarkResult(
                f"Soft Threshold (batch={batch_size})",
                pytorch_time_ms,
                cuda_time_ms,
                batch_size,
                self.device,
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_truth_degree(
        self, batch_sizes: List[int] = None
    ) -> List[BenchmarkResult]:
        """Benchmark truth degree computation.

        Args:
            batch_sizes: Batch sizes to test

        Returns:
            List of benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [32, 64, 128, 256, 512, 1024]

        results = []

        for batch_size in batch_sizes:
            canonical_states = torch.randn(
                batch_size, 8, device=self.device
            )

            # PyTorch baseline
            pytorch_verifier = BatchedPolicyVerifier(device=self.device)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                vec_part = canonical_states[:, 1:4]
                x_t = canonical_states[:, 3]
                norms = torch.linalg.norm(vec_part, dim=1)
                _ = torch.abs(x_t) / (norms + 1e-8)
            torch.cuda.synchronize()
            pytorch_time_ms = (time.time() - start) * 10

            # CUDA optimized
            cuda_verifier = CUDABatchedPolicyVerifier(device=self.device)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = cuda_verifier._compute_truth_degrees_cuda(
                    canonical_states
                )
            torch.cuda.synchronize()
            cuda_time_ms = (time.time() - start) * 10

            result = BenchmarkResult(
                f"Truth Degree (batch={batch_size})",
                pytorch_time_ms,
                cuda_time_ms,
                batch_size,
                self.device,
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_batch_policies(
        self, batch_sizes: List[int] = None
    ) -> List[BenchmarkResult]:
        """Benchmark batch policy verification.

        Args:
            batch_sizes: Batch sizes to test

        Returns:
            List of benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [32, 64, 128, 256]

        results = []

        for batch_size in batch_sizes:
            fact_values = {
                "age": torch.randn(batch_size, device=self.device),
                "budget": torch.randn(batch_size, device=self.device),
            }

            policy_specs = [
                {
                    "name": "policy1",
                    "conditions": [
                        {"subject": "age", "threshold": 0.5, "mode": "soft"},
                        {
                            "subject": "budget",
                            "threshold": 0.5,
                            "mode": "soft",
                        },
                    ],
                    "operator": "and",
                }
            ]

            # PyTorch baseline
            pytorch_verifier = BatchedPolicyVerifier(device=self.device)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _ = pytorch_verifier.evaluate_batch_policies(
                    fact_values, policy_specs
                )
            torch.cuda.synchronize()
            pytorch_time_ms = (time.time() - start) * 100

            # CUDA optimized
            cuda_verifier = CUDABatchedPolicyVerifier(
                device=self.device, enable_profiling=True
            )
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _ = cuda_verifier.evaluate_batch_policies_cuda(
                    fact_values, policy_specs
                )
            torch.cuda.synchronize()
            cuda_time_ms = (time.time() - start) * 100

            result = BenchmarkResult(
                f"Batch Policies (batch={batch_size})",
                pytorch_time_ms,
                cuda_time_ms,
                batch_size,
                self.device,
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_scaling(self) -> Dict[str, float]:
        """Benchmark scaling behavior with increasing batch size.

        Tests batch sizes from 32 to 2048 to measure throughput scaling.

        Returns:
            Dict mapping batch size to throughput (facts/sec)
        """
        scaling = {}

        for batch_size in [32, 64, 128, 256, 512, 1024, 2048]:
            fact_values = {
                "x": torch.randn(batch_size, device=self.device),
            }

            policy_specs = [
                {
                    "name": "policy",
                    "conditions": [
                        {"subject": "x", "threshold": 0.5, "mode": "soft"}
                    ],
                    "operator": "and",
                }
            ]

            cuda_verifier = CUDABatchedPolicyVerifier(device=self.device)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(5):
                _ = cuda_verifier.evaluate_batch_policies_cuda(
                    fact_values, policy_specs
                )
            torch.cuda.synchronize()
            elapsed_ms = (time.time() - start) * 200

            throughput = (batch_size / elapsed_ms) * 1000
            scaling[f"{batch_size}"] = throughput

        return scaling

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 120)
        print("CUDA Optimization Benchmark Results")
        print("=" * 120)

        for result in self.results:
            print(result)

        print("\n" + "-" * 120)
        print("Summary Statistics")
        print("-" * 120)

        speedups = [r.speedup for r in self.results]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

        pytorch_throughputs = [r.pytorch_throughput for r in self.results]
        cuda_throughputs = [r.cuda_throughput for r in self.results]

        avg_pytorch = (
            sum(pytorch_throughputs) / len(pytorch_throughputs)
            if pytorch_throughputs
            else 0.0
        )
        avg_cuda = (
            sum(cuda_throughputs) / len(cuda_throughputs)
            if cuda_throughputs
            else 0.0
        )

        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"PyTorch Avg Throughput: {avg_pytorch:.0f} facts/sec")
        print(f"CUDA Avg Throughput: {avg_cuda:.0f} facts/sec")

        print("\n" + "=" * 120 + "\n")


def run_benchmarks():
    """Run full benchmark suite."""
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU (expect slower results)")
        device = "cpu"
    else:
        device = "cuda"

    suite = CUDABenchmarkSuite(device=device)

    print("Running Hard Threshold benchmarks...")
    suite.benchmark_hard_threshold()

    print("Running Soft Threshold benchmarks...")
    suite.benchmark_soft_threshold()

    print("Running Truth Degree benchmarks...")
    suite.benchmark_truth_degree()

    print("Running Batch Policy benchmarks...")
    suite.benchmark_batch_policies()

    print("\nRunning Scaling analysis...")
    scaling_results = suite.benchmark_scaling()
    print("Scaling Results (batch_size -> throughput):")
    for batch_size, throughput in scaling_results.items():
        print(f"  Batch {batch_size:4s}: {throughput:7.0f} facts/sec")

    suite.print_summary()


if __name__ == "__main__":
    run_benchmarks()
