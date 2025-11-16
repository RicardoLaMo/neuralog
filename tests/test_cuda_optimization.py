"""Tests for CUDA Optimization Components.

Tests cover:
- CUDA kernel initialization and configuration
- Threshold evaluation kernels
- Memory optimization
- Benchmarking suite
- H200 performance monitoring
"""

import pytest
import torch

from neuralog.symbolic.geometric_algebra.cuda_batched_verifier import (
    CUDABatchedPolicyVerifier,
)
from neuralog.symbolic.geometric_algebra.cuda_kernels import (
    CUDAGraphScheduler,
    CUDAKernelConfig,
    FusedThresholdKernel,
    H200PerformanceMonitor,
    MemoryOptimizer,
    TruthDegreeKernel,
)


class TestCUDAKernelConfig:
    """Tests for CUDA kernel configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = CUDAKernelConfig()
        assert config.block_size == 256
        assert config.use_shared_memory is True
        assert config.use_fp8 is False
        assert config.enable_graphs is True

    def test_h200_optimized_config(self):
        """Test H200-optimized configuration."""
        config = CUDAKernelConfig.h200_optimized()
        assert config.block_size == 256
        assert config.use_fp8 is True  # H200 has FP8 support
        assert config.enable_graphs is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CUDAKernelConfig(
            block_size=512, use_fp8=True, use_shared_memory=False
        )
        assert config.block_size == 512
        assert config.use_fp8 is True
        assert config.use_shared_memory is False


class TestFusedThresholdKernel:
    """Tests for fused threshold evaluation kernel."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CUDAKernelConfig()
        self.kernel = FusedThresholdKernel(self.config)

    def test_kernel_initialization(self):
        """Test kernel initialization."""
        assert self.kernel.config is not None
        assert self.kernel._cuda_graph is None

    def test_hard_threshold_launch(self):
        """Test hard threshold kernel launch."""
        values = torch.tensor([60.0, 65.0, 70.0, 75.0])
        thresholds = torch.tensor([65.0, 65.0, 65.0, 65.0])
        output = torch.empty_like(values)

        self.kernel.launch_hard_threshold(values, thresholds, output)

        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        assert torch.allclose(output, expected)

    def test_soft_threshold_launch(self):
        """Test soft threshold kernel launch."""
        values = torch.tensor([65.0, 65.0])
        thresholds = torch.tensor([65.0, 65.0])
        output = torch.empty_like(values)

        self.kernel.launch_soft_threshold(values, thresholds, 20.0, output)

        # sigmoid(0) ≈ 0.5
        assert torch.allclose(output, torch.tensor([0.5, 0.5]), atol=0.01)

    def test_margin_threshold_launch(self):
        """Test margin threshold kernel launch."""
        values = torch.tensor([65.5])
        thresholds = torch.tensor([65.0])
        output = torch.empty_like(values)

        self.kernel.launch_margin_threshold(values, thresholds, 20.0, 0.5, output)

        # Result should be between 0 and 1
        assert 0.0 <= output.item() <= 1.0

    def test_batched_threshold_launch(self):
        """Test batched threshold evaluation with mixed modes."""
        values = torch.randn(4)
        thresholds = torch.randn(4)
        modes = ["hard", "soft", "margin", "soft"]

        output = self.kernel.launch_batched(
            values, thresholds, 20.0, 0.5, modes
        )

        assert output.shape == values.shape


class TestTruthDegreeKernel:
    """Tests for truth degree computation kernel."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kernel = TruthDegreeKernel()

    def test_truth_degree_computation(self):
        """Test truth degree computation."""
        canonical_states = torch.tensor(
            [
                [0.0, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.6, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        truths = self.kernel.compute_truth_degrees(canonical_states)

        assert truths.shape == (2,)
        assert (truths >= 0.0).all() and (truths <= 1.0).all()

    def test_truth_degree_batch(self):
        """Test batched truth degree computation."""
        batch_size = 100
        canonical_states = torch.randn(batch_size, 8)

        truths = self.kernel.compute_truth_degrees(canonical_states)

        assert truths.shape == (batch_size,)
        assert (truths >= 0.0).all() and (truths <= 1.0).all()


class TestMemoryOptimizer:
    """Tests for memory optimization utilities."""

    def test_coalesce_tensor(self):
        """Test tensor coalescing."""
        tensor = torch.randn(100, 100).t()  # Non-contiguous
        assert not tensor.is_contiguous()

        coalesced = MemoryOptimizer.coalesce_tensor(tensor)
        assert coalesced.is_contiguous()

    def test_already_coalesced(self):
        """Test coalescing of already-coalesced tensor."""
        tensor = torch.randn(100, 100)
        assert tensor.is_contiguous()

        coalesced = MemoryOptimizer.coalesce_tensor(tensor)
        assert coalesced.is_contiguous()

    def test_allocate_pinned(self):
        """Test pinned memory allocation."""
        shape = (1000, 100)
        pinned = MemoryOptimizer.allocate_pinned(shape)

        assert pinned.shape == shape
        assert pinned.pin_memory() or not torch.cuda.is_available()

    def test_prefetch_to_gpu(self):
        """Test GPU prefetch."""
        tensor = torch.randn(100, 100)

        if torch.cuda.is_available():
            prefetched = MemoryOptimizer.prefetch_to_gpu(tensor)
            assert prefetched.is_cuda
        else:
            prefetched = MemoryOptimizer.prefetch_to_gpu(tensor)
            assert prefetched is tensor


class TestCUDAGraphScheduler:
    """Tests for CUDA graph scheduling."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = CUDAGraphScheduler()
        assert len(scheduler.graphs) == 0

    def test_clear_graphs(self):
        """Test graph clearing."""
        scheduler = CUDAGraphScheduler()
        scheduler.graphs["test"] = "dummy_graph"
        assert "test" in scheduler.graphs

        scheduler.clear_graphs()
        assert len(scheduler.graphs) == 0


class TestH200PerformanceMonitor:
    """Tests for H200 performance monitoring."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = H200PerformanceMonitor()
        assert "kernel_times" in monitor.metrics
        assert "memory_bandwidth" in monitor.metrics
        assert "occupancy" in monitor.metrics

    def test_record_kernel_time(self):
        """Test kernel time recording."""
        monitor = H200PerformanceMonitor()
        monitor.record_kernel_time("test_kernel", 10.5)

        assert len(monitor.metrics["kernel_times"]) == 1
        assert monitor.metrics["kernel_times"][0] == ("test_kernel", 10.5)

    def test_record_memory_bandwidth(self):
        """Test memory bandwidth recording."""
        monitor = H200PerformanceMonitor()
        monitor.record_memory_bandwidth(1e9, 1000)  # 1GB in 1 second

        assert len(monitor.metrics["memory_bandwidth"]) == 1
        assert abs(monitor.metrics["memory_bandwidth"][0] - 1.0) < 0.1

    def test_record_occupancy(self):
        """Test occupancy recording."""
        monitor = H200PerformanceMonitor()
        monitor.record_occupancy(85.5)

        assert len(monitor.metrics["occupancy"]) == 1
        assert monitor.metrics["occupancy"][0] == 85.5

    def test_summary(self):
        """Test summary generation."""
        monitor = H200PerformanceMonitor()
        monitor.record_kernel_time("kernel1", 10.0)
        monitor.record_memory_bandwidth(1e9, 1000)
        monitor.record_occupancy(80.0)

        summary = monitor.summary()

        assert "H200 Performance Summary" in summary
        assert "kernel" in summary.lower() or "time" in summary.lower()


class TestCUDABatchedPolicyVerifier:
    """Tests for CUDA-optimized batched verifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = CUDABatchedPolicyVerifier(
            device="cpu", enable_profiling=True
        )

    def test_initialization(self):
        """Test verifier initialization."""
        assert self.verifier.threshold_kernel is not None
        assert self.verifier.truth_kernel is not None
        assert self.verifier.monitor is not None

    def test_fused_threshold_eval_cuda_hard(self):
        """Test CUDA-optimized hard threshold."""
        values = torch.tensor([60.0, 65.0, 70.0, 75.0])
        thresholds = torch.tensor([65.0, 65.0, 65.0, 65.0])

        result = self.verifier._fused_threshold_eval_cuda(
            values, thresholds, "hard"
        )

        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_fused_threshold_eval_cuda_soft(self):
        """Test CUDA-optimized soft threshold."""
        values = torch.tensor([65.0, 65.0])
        thresholds = torch.tensor([65.0, 65.0])

        result = self.verifier._fused_threshold_eval_cuda(
            values, thresholds, "soft"
        )

        assert torch.allclose(result, torch.tensor([0.5, 0.5]), atol=0.01)

    def test_compute_truth_degrees_cuda(self):
        """Test CUDA-optimized truth degree computation."""
        canonical_states = torch.randn(32, 8)

        truths = self.verifier._compute_truth_degrees_cuda(canonical_states)

        assert truths.shape == (32,)
        assert (truths >= 0.0).all() and (truths <= 1.0).all()

    def test_evaluate_batch_numeric_cuda(self):
        """Test CUDA-optimized batch numeric evaluation."""
        subjects = torch.arange(4)
        predicates = torch.zeros(4)
        values = torch.tensor([60.0, 65.0, 70.0, 75.0])
        thresholds = torch.tensor([65.0, 65.0, 65.0, 65.0])

        states, truths, losses = self.verifier.evaluate_batch_numeric_cuda(
            subjects, predicates, values, thresholds, "soft"
        )

        assert states.shape == (4, 8)
        assert truths.shape == (4,)
        assert losses.shape == (4,)

    def test_performance_monitoring(self):
        """Test performance monitoring."""
        assert self.verifier.monitor is not None

        self.verifier.monitor.record_kernel_time("test", 10.0)
        summary = self.verifier.get_performance_summary()

        assert "H200" in summary or "Performance" in summary

    def test_cuda_graphs_enabled(self):
        """Test CUDA graphs enabling."""
        self.verifier.enable_cuda_graphs()
        assert self.verifier.graph_scheduler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
