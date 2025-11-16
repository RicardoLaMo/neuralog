"""Unified NeuraLog Geometric Algebra Reasoning System.

Orchestrates policy verification, counterfactual reasoning, LLM integration,
and metrics collection into a cohesive end-to-end system.

Supports:
- Streaming fact evaluation from LLM
- Async policy verification on GPU
- Counterfactual explanation generation
- Real-time metrics collection and monitoring
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from neuralog.symbolic.geometric_algebra.batched_verifier import (
    BatchedPolicyVerifier,
    BatchVerificationResult,
)
from neuralog.symbolic.geometric_algebra.counterfactual_reasoner import (
    CounterfactualExplanation,
    CounterfactualReasoner,
)
from neuralog.symbolic.geometric_algebra.verification_metrics import (
    MetricsCollector,
    VerificationMetrics,
)


@dataclass
class PolicyVerificationResult:
    """Complete verification result for a fact."""

    fact_id: str
    extracted_values: Dict[str, float]
    verification: Dict  # From BatchVerificationResult
    counterfactual: Optional[CounterfactualExplanation] = None
    is_consistent: bool = False
    confidence_score: float = 0.0


@dataclass
class SystemConfig:
    """Configuration for NeuraLogGASystem."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    use_fp8: bool = False
    enable_counterfactuals: bool = True
    enable_metrics: bool = True
    max_concurrent_policies: int = 5
    timeout_seconds: float = 30.0


class NeuraLogGASystem(torch.nn.Module):
    """Unified Geometric Algebra reasoning system for NeuraLog.

    Orchestrates:
    1. LLM fact extraction
    2. Policy verification with GA truth degrees
    3. Counterfactual explanation generation
    4. Metrics collection and monitoring

    Attributes:
        verifier: BatchedPolicyVerifier for fast GPU evaluation
        reasoner: CounterfactualReasoner for explanation generation
        metrics: MetricsCollector for performance tracking
        policies: Registered policy specifications
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize NeuraLogGASystem.

        Args:
            config: System configuration. Defaults to auto-detected config.
        """
        super().__init__()
        self.config = config or SystemConfig()

        # Core components
        self.verifier = BatchedPolicyVerifier(
            device=self.config.device,
            use_fp8=self.config.use_fp8,
        )
        self.reasoner = CounterfactualReasoner(device=self.config.device)
        self.metrics = MetricsCollector(device=self.config.device)

        # Policy registry
        self.policies: Dict[str, Dict] = {}
        self.policy_order: List[str] = []

        # Async tracking
        self._pending_tasks = {}

    def register_policy(self, name: str, spec: Dict) -> None:
        """Register a policy for verification.

        Args:
            name: Policy identifier
            spec: Policy specification with conditions, operator, etc.
        """
        self.policies[name] = spec
        if name not in self.policy_order:
            self.policy_order.append(name)

    def unregister_policy(self, name: str) -> None:
        """Unregister a policy.

        Args:
            name: Policy identifier
        """
        if name in self.policies:
            del self.policies[name]
            self.policy_order.remove(name)

    def verify_facts(
        self,
        facts: List[Dict[str, float]],
        policy_names: Optional[List[str]] = None,
        generate_counterfactuals: bool = None,
    ) -> List[PolicyVerificationResult]:
        """Verify a batch of facts against policies.

        Args:
            facts: List of fact value dicts
            policy_names: Specific policies to use. All if None.
            generate_counterfactuals: Override config setting

        Returns:
            List of PolicyVerificationResult objects
        """
        if not facts:
            return []

        # Determine which policies to use
        if policy_names is None:
            policies_to_check = self.policies
        else:
            policies_to_check = {
                name: self.policies[name]
                for name in policy_names
                if name in self.policies
            }

        if not policies_to_check:
            raise ValueError("No policies registered or specified")

        # Convert facts to batched tensors
        all_subjects = set()
        for fact in facts:
            all_subjects.update(fact.keys())

        fact_values = {}
        for subject in all_subjects:
            values = []
            for fact in facts:
                values.append(fact.get(subject, 0.0))
            fact_values[subject] = torch.tensor(
                values, dtype=torch.float32, device=self.config.device
            )

        # Run batched verification
        policy_specs = list(policies_to_check.values())
        batch_result = self.verifier.evaluate_batch_policies(
            fact_values, policy_specs
        )

        # Process results
        results = []
        generate_cf = (
            generate_counterfactuals
            if generate_counterfactuals is not None
            else self.config.enable_counterfactuals
        )

        for i, fact in enumerate(facts):
            # Find corresponding results
            fact_results = [r for r in batch_result.results if r.fact_id == f"fact_{i}"]

            # Aggregate verification
            is_consistent = all(r.satisfies_policy for r in fact_results)
            mean_confidence = (
                sum(r.truth_degree for r in fact_results) / len(fact_results)
                if fact_results
                else 0.0
            )

            # Generate counterfactual if enabled
            counterfactual = None
            if generate_cf and not is_consistent and policies_to_check:
                first_policy = list(policies_to_check.values())[0]
                try:
                    counterfactual = self.reasoner.find_counterfactual(
                        fact, first_policy
                    )
                except Exception:
                    # Silently skip if optimization fails
                    pass

            # Update metrics
            if self.config.enable_metrics:
                self.metrics.total_facts += 1
                self.metrics.update_truth_degree(mean_confidence)
                for r in fact_results:
                    self.metrics.update_loss(
                        r.implication_loss,
                        r.fp_loss,
                        r.fn_loss,
                    )

            results.append(
                PolicyVerificationResult(
                    fact_id=f"fact_{i}",
                    extracted_values=fact,
                    verification=batch_result.__dict__,
                    counterfactual=counterfactual,
                    is_consistent=is_consistent,
                    confidence_score=mean_confidence,
                )
            )

        return results

    async def verify_facts_async(
        self,
        facts: List[Dict[str, float]],
        policy_names: Optional[List[str]] = None,
    ) -> List[PolicyVerificationResult]:
        """Async verification for high-throughput scenarios.

        Args:
            facts: List of fact dicts
            policy_names: Specific policies to use

        Returns:
            List of PolicyVerificationResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.verify_facts,
            facts,
            policy_names,
        )

    def stream_verify(
        self,
        fact_generator,
        policy_names: Optional[List[str]] = None,
    ):
        """Stream verification for large datasets.

        Args:
            fact_generator: Generator yielding fact dicts
            policy_names: Specific policies to use

        Yields:
            PolicyVerificationResult objects
        """
        batch = []
        for fact in fact_generator:
            batch.append(fact)

            if len(batch) >= self.config.batch_size:
                results = self.verify_facts(batch, policy_names)
                for result in results:
                    yield result
                batch = []

        # Process remaining
        if batch:
            results = self.verify_facts(batch, policy_names)
            for result in results:
                yield result

    def get_metrics_summary(self) -> VerificationMetrics:
        """Get current metrics summary.

        Returns:
            VerificationMetrics object
        """
        return self.metrics.compute_metrics(num_policies=len(self.policies))

    def log_metrics(self) -> str:
        """Get formatted metrics log.

        Returns:
            Formatted metrics summary
        """
        return self.metrics.log_summary()

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics.reset()

    def export_metrics_json(self) -> Dict:
        """Export metrics as JSON-serializable dict.

        Returns:
            Metrics dictionary
        """
        metrics = self.get_metrics_summary()
        return {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "specificity": metrics.specificity,
            "mean_truth_degree": metrics.mean_truth_degree,
            "std_truth_degree": metrics.std_truth_degree,
            "satisfaction_rate": metrics.satisfaction_rate,
            "mean_implication_loss": metrics.mean_implication_loss,
            "mean_fp_loss": metrics.mean_fp_loss,
            "mean_fn_loss": metrics.mean_fn_loss,
            "latency_p50_ms": metrics.latency.p50,
            "latency_p95_ms": metrics.latency.p95,
            "latency_p99_ms": metrics.latency.p99,
            "latency_mean_ms": metrics.latency.mean,
            "throughput_facts_per_sec": metrics.throughput.facts_per_second,
            "num_facts": metrics.num_facts,
            "num_policies": metrics.num_policies,
            "device": metrics.device,
        }

    def to(self, device):
        """Move system to device.

        Args:
            device: PyTorch device

        Returns:
            Self
        """
        super().to(device)
        self.config.device = str(device)
        self.verifier = self.verifier.to(device)
        self.reasoner = self.reasoner.to(device)
        return self

    def forward(
        self,
        facts: List[Dict[str, float]],
        policy_names: Optional[List[str]] = None,
    ) -> List[PolicyVerificationResult]:
        """Forward pass: verify facts against policies.

        Args:
            facts: List of fact dicts
            policy_names: Specific policies to check

        Returns:
            List of PolicyVerificationResult
        """
        return self.verify_facts(facts, policy_names)

    def summary(self) -> str:
        """Print system summary.

        Returns:
            Formatted summary string
        """
        lines = [
            "NeuraLog Geometric Algebra System",
            "=" * 50,
            f"Device: {self.config.device}",
            f"Batch Size: {self.config.batch_size}",
            f"FP8 Precision: {self.config.use_fp8}",
            f"Policies Registered: {len(self.policies)}",
            "",
            "Registered Policies:",
        ]

        for name in self.policy_order:
            policy = self.policies[name]
            num_conditions = len(policy.get("conditions", []))
            lines.append(f"  - {name}: {num_conditions} conditions")

        return "\n".join(lines)
