"""Distributed Inference for Multi-GPU Policy Verification.

Implements multi-GPU policy evaluation using PyTorch DDP for scaling
to multiple GPUs and nodes.

Features:
- Data parallel policy verification
- Automatic policy sharding
- Cross-GPU metric aggregation
- Fault tolerance and checkpointing
- Efficient communication patterns
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from neuralog.symbolic.geometric_algebra.cuda_batched_verifier import (
    CUDABatchedPolicyVerifier,
)


class DistributedPolicyVerifier(nn.Module):
    """Multi-GPU distributed policy verifier.

    Uses PyTorch DDP for data parallelism across GPUs.
    Each GPU handles a portion of the facts.

    Attributes:
        verifier: Base CUDABatchedPolicyVerifier
        rank: Process rank in distributed group
        world_size: Total number of processes
        device: Current device
    """

    def __init__(
        self,
        base_verifier: Optional[CUDABatchedPolicyVerifier] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Initialize distributed verifier.

        Args:
            base_verifier: Base verifier to wrap
            rank: Process rank
            world_size: Number of processes
        """
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )

        if base_verifier is None:
            base_verifier = CUDABatchedPolicyVerifier(device=str(self.device))

        self.verifier = base_verifier.to(self.device)

        # Wrap with DDP if multi-GPU
        if world_size > 1 and torch.cuda.is_available():
            self.verifier = DDP(
                self.verifier,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )

    def forward(
        self,
        fact_values: Dict[str, torch.Tensor],
        policy_specs: List[Dict],
    ) -> Dict:
        """Forward pass with distributed verification.

        Args:
            fact_values: Dict mapping subjects to value tensors
            policy_specs: List of policy specifications

        Returns:
            Dict with verification results
        """
        # Move data to device
        fact_values = {
            k: v.to(self.device) for k, v in fact_values.items()
        }

        # Verify on local GPU
        if isinstance(self.verifier, DDP):
            result = self.verifier.module.evaluate_batch_policies_cuda(
                fact_values, policy_specs
            )
        else:
            result = self.verifier.evaluate_batch_policies_cuda(
                fact_values, policy_specs
            )

        return result

    def reduce_metric(self, metric: torch.Tensor) -> torch.Tensor:
        """Reduce metric across all processes.

        Args:
            metric: Local metric value

        Returns:
            Averaged metric across all processes
        """
        if self.world_size == 1:
            return metric

        metric = metric.to(self.device)
        torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.AVG)
        return metric


class PolicyShardingStrategy:
    """Strategy for sharding policies across GPUs.

    Distributes policies to different GPUs for better parallelism.
    """

    def __init__(self, num_gpus: int, policies: List[Dict]):
        """Initialize policy sharding.

        Args:
            num_gpus: Number of GPUs
            policies: List of policies to shard
        """
        self.num_gpus = num_gpus
        self.policies = policies
        self.shards = self._compute_shards()

    def _compute_shards(self) -> List[List[Dict]]:
        """Compute policy shards for each GPU.

        Returns:
            List of policy lists, one per GPU
        """
        shards = [[] for _ in range(self.num_gpus)]

        for i, policy in enumerate(self.policies):
            gpu_id = i % self.num_gpus
            shards[gpu_id].append(policy)

        return shards

    def get_shard_for_gpu(self, gpu_id: int) -> List[Dict]:
        """Get policies assigned to GPU.

        Args:
            gpu_id: GPU device ID

        Returns:
            List of policies for this GPU
        """
        if gpu_id >= self.num_gpus:
            raise ValueError(f"GPU {gpu_id} >= {self.num_gpus}")
        return self.shards[gpu_id]

    def get_all_shards(self) -> List[List[Dict]]:
        """Get all policy shards.

        Returns:
            All shards
        """
        return self.shards


class DistributedMetricsAggregator:
    """Aggregates metrics across all distributed processes.

    Collects and combines metrics from multiple GPUs.
    """

    def __init__(self, rank: int, world_size: int):
        """Initialize metrics aggregator.

        Args:
            rank: Process rank
            world_size: Number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.local_metrics = {}
        self.global_metrics = {}

    def record_local_metric(self, name: str, value: float) -> None:
        """Record a local metric.

        Args:
            name: Metric name
            value: Metric value
        """
        self.local_metrics[name] = value

    def synchronize_metrics(self) -> Dict[str, float]:
        """Synchronize metrics across all processes.

        Returns:
            Dict with synchronized (averaged) metrics
        """
        if self.world_size == 1:
            return self.local_metrics

        # For simplicity, gather all metrics to rank 0
        metrics_list = [None] * self.world_size

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            metrics_tensor = torch.tensor(
                list(self.local_metrics.values()),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )

            # All gather to rank 0
            gathered = [
                torch.zeros_like(metrics_tensor)
                for _ in range(self.world_size)
            ]

            torch.distributed.all_gather(gathered, metrics_tensor)

            if self.rank == 0:
                # Average across processes
                all_metrics = torch.stack(gathered).mean(dim=0)
                self.global_metrics = {
                    name: all_metrics[i].item()
                    for i, name in enumerate(self.local_metrics.keys())
                }
            else:
                self.global_metrics = self.local_metrics
        else:
            self.global_metrics = self.local_metrics

        return self.global_metrics

    def get_global_metrics(self) -> Dict[str, float]:
        """Get synchronized global metrics.

        Returns:
            Global metrics (average across processes)
        """
        return self.global_metrics


class DistributedInferencePipeline:
    """End-to-end distributed inference pipeline.

    Orchestrates distributed policy verification with metrics
    collection and synchronization.

    Attributes:
        rank: Process rank
        world_size: Number of processes
        verifier: Distributed verifier
        policy_sharding: Policy sharding strategy
        metrics_aggregator: Metrics aggregator
    """

    def __init__(
        self,
        num_gpus: int = 1,
        policies: Optional[List[Dict]] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialize distributed pipeline.

        Args:
            num_gpus: Number of GPUs to use
            policies: List of policies to shard
            checkpoint_dir: Directory for checkpoints
        """
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.num_gpus = num_gpus
        self.checkpoint_dir = checkpoint_dir

        # Initialize components
        self.verifier = DistributedPolicyVerifier(
            rank=self.rank,
            world_size=self.world_size,
        )

        self.policy_sharding = (
            PolicyShardingStrategy(self.world_size, policies or [])
            if policies
            else None
        )

        self.metrics_aggregator = DistributedMetricsAggregator(
            self.rank, self.world_size
        )

    def verify_facts_distributed(
        self,
        fact_values: Dict[str, torch.Tensor],
        policy_specs: List[Dict],
    ) -> Dict:
        """Verify facts with distributed processing.

        Args:
            fact_values: Dict of fact value tensors
            policy_specs: List of policies

        Returns:
            Dict with verification results
        """
        # Shard policies across GPUs if configured
        if self.policy_sharding:
            policies_for_this_gpu = self.policy_sharding.get_shard_for_gpu(
                self.rank
            )
        else:
            policies_for_this_gpu = policy_specs

        # Verify on local GPU
        local_result = self.verifier(fact_values, policies_for_this_gpu)

        # Record local metrics
        self.metrics_aggregator.record_local_metric(
            "throughput", local_result.throughput
        )
        self.metrics_aggregator.record_local_metric(
            "mean_truth_degree", local_result.mean_truth_degree
        )
        self.metrics_aggregator.record_local_metric(
            "satisfaction_rate", local_result.satisfaction_rate
        )

        # Synchronize metrics
        global_metrics = self.metrics_aggregator.synchronize_metrics()

        return {
            "local_result": local_result,
            "global_metrics": global_metrics,
        }

    def checkpoint(self, step: int) -> Optional[str]:
        """Save checkpoint.

        Args:
            step: Training/evaluation step

        Returns:
            Checkpoint path or None
        """
        if self.checkpoint_dir is None or self.rank != 0:
            return None

        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_step_{step}.pt"
        )

        # Save verifier state
        if hasattr(self.verifier.verifier, "module"):
            state_dict = self.verifier.verifier.module.state_dict()
        else:
            state_dict = self.verifier.verifier.state_dict()

        torch.save(
            {
                "step": step,
                "verifier_state": state_dict,
                "rank": self.rank,
            },
            checkpoint_path,
        )

        return checkpoint_path

    def restore_checkpoint(self, checkpoint_path: str) -> int:
        """Restore from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Step number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.verifier.device)

        if hasattr(self.verifier.verifier, "module"):
            self.verifier.verifier.module.load_state_dict(
                checkpoint["verifier_state"]
            )
        else:
            self.verifier.verifier.load_state_dict(checkpoint["verifier_state"])

        return checkpoint["step"]


def init_distributed_training(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed training.

    Args:
        rank: Process rank
        world_size: Number of processes
        backend: Communication backend ("nccl" or "gloo")

    Raises:
        RuntimeError: If distributed setup fails
    """
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    if not torch.cuda.is_available():
        backend = "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed():
    """Clean up distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
