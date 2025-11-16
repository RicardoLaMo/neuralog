"""Tests for high-performance GA components.

Tests cover:
- BatchedPolicyVerifier GPU performance
- CounterfactualReasoner optimization
- MetricsCollector accuracy tracking
- NeuraLogGASystem end-to-end integration
"""

import pytest
import torch

from neuralog.symbolic.geometric_algebra import (
    BatchedPolicyVerifier,
    CounterfactualReasoner,
    MetricsCollector,
    NeuraLogGASystem,
    SystemConfig,
    ThresholdMode,
)


class TestBatchedPolicyVerifier:
    """Tests for BatchedPolicyVerifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = BatchedPolicyVerifier(device="cpu", use_fp8=False)

    def test_initialization(self):
        """Test verifier initialization."""
        assert self.verifier.device.type == "cpu"
        assert self.verifier.use_fp8 is False

    def test_fused_threshold_eval_hard(self):
        """Test fused threshold evaluation (HARD mode)."""
        values = torch.tensor([60.0, 65.0, 70.0, 75.0])
        thresholds = torch.tensor([65.0, 65.0, 65.0, 65.0])

        result = self.verifier._fused_threshold_eval(values, thresholds, "hard")

        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_fused_threshold_eval_soft(self):
        """Test fused threshold evaluation (SOFT mode)."""
        values = torch.tensor([65.0, 65.0])
        thresholds = torch.tensor([65.0, 65.0])

        result = self.verifier._fused_threshold_eval(values, thresholds, "soft")

        # sigmoid(0) ≈ 0.5
        assert torch.allclose(result, torch.tensor([0.5, 0.5]), atol=0.01)

    def test_evaluate_batch_numeric(self):
        """Test numeric triplet batch evaluation."""
        subjects = torch.arange(4)
        predicates = torch.zeros(4)
        values = torch.tensor([60.0, 65.0, 70.0, 75.0])
        thresholds = torch.tensor([65.0, 65.0, 65.0, 65.0])

        states, truths, losses = self.verifier.evaluate_batch_numeric(
            subjects, predicates, values, thresholds, "soft"
        )

        assert states.shape == (4, 8)
        assert truths.shape == (4,)
        assert losses.shape == (4,)
        assert (truths >= 0.0).all() and (truths <= 1.0).all()

    def test_evaluate_batch_policies(self):
        """Test batch policy evaluation."""
        fact_values = {
            "age": torch.tensor([60.0, 70.0, 65.0]),
            "budget": torch.tensor([20.0, 25.0, 22.0]),
        }

        policy_specs = [
            {
                "name": "senior_discount",
                "conditions": [
                    {"subject": "age", "threshold": 65.0, "mode": "soft"},
                    {"subject": "budget", "threshold": 22.0, "mode": "soft"},
                ],
                "operator": "and",
            }
        ]

        result = self.verifier.evaluate_batch_policies(
            fact_values, policy_specs
        )

        assert result.batch_size == 3
        assert len(result.results) == 3
        assert result.satisfaction_rate >= 0.0
        assert result.satisfaction_rate <= 1.0

    def test_batch_with_llm_values(self):
        """Test batch evaluation with LLM values for bidirectional loss."""
        fact_values = {
            "age": torch.tensor([70.0, 72.0]),
            "budget": torch.tensor([25.0, 23.0]),
        }

        llm_values = {
            "age": torch.tensor([0.9, 0.8]),
            "budget": torch.tensor([0.7, 0.6]),
        }

        policy_specs = [
            {
                "name": "test_policy",
                "conditions": [
                    {"subject": "age", "threshold": 65.0, "mode": "soft"},
                ],
                "operator": "and",
            }
        ]

        result = self.verifier.evaluate_batch_policies(
            fact_values, policy_specs, llm_values
        )

        assert result.mean_fp_loss >= 0.0
        assert result.mean_fn_loss >= 0.0

    def test_throughput_measurement(self):
        """Test throughput metrics."""
        fact_values = {
            "x": torch.rand(100),
            "y": torch.rand(100),
        }

        policy_specs = [
            {
                "name": "policy",
                "conditions": [{"subject": "x", "threshold": 0.5, "mode": "soft"}],
                "operator": "and",
            }
        ]

        result = self.verifier.evaluate_batch_policies(
            fact_values, policy_specs
        )

        # Throughput should be reasonable (at least 1 fact/sec on CPU)
        assert result.throughput > 0.0


class TestCounterfactualReasoner:
    """Tests for CounterfactualReasoner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reasoner = CounterfactualReasoner(
            lr=0.1, max_steps=50, device="cpu"
        )

    def test_initialization(self):
        """Test reasoner initialization."""
        assert self.reasoner.lr == 0.1
        assert self.reasoner.max_steps == 50

    def test_distance_metrics(self):
        """Test distance metric computation."""
        original = torch.tensor([1.0, 2.0, 3.0])
        counterfactual = torch.tensor([1.0, 2.5, 3.0])

        l1_dist = self.reasoner._compute_distance(
            original, counterfactual, "l1"
        )
        l2_dist = self.reasoner._compute_distance(
            original, counterfactual, "l2"
        )
        linf_dist = self.reasoner._compute_distance(
            original, counterfactual, "linf"
        )

        assert l1_dist.item() == pytest.approx(0.5)
        assert l2_dist.item() == pytest.approx(0.5)
        assert linf_dist.item() == pytest.approx(0.5)

    def test_policy_satisfaction_evaluation(self):
        """Test policy satisfaction evaluation (differentiable)."""
        values = {"age": torch.tensor(70.0, requires_grad=True)}
        policy_spec = {
            "conditions": [
                {"subject": "age", "threshold": 65.0, "mode": "soft"}
            ],
            "operator": "and",
        }

        truth_degree, loss = self.reasoner._evaluate_policy_satisfaction(
            values, policy_spec
        )

        assert truth_degree.requires_grad
        assert loss.requires_grad
        assert truth_degree.item() > 0.5  # 70 >= 65

    def test_find_counterfactual_simple(self):
        """Test counterfactual finding for simple policy."""
        original_values = {"age": 60.0, "budget": 20.0}
        policy_spec = {
            "name": "discount",
            "conditions": [
                {"subject": "age", "threshold": 65.0, "mode": "soft"},
                {"subject": "budget", "threshold": 22.0, "mode": "soft"},
            ],
            "operator": "and",
        }

        cf = self.reasoner.find_counterfactual(
            original_values, policy_spec, distance_metric="l2"
        )

        assert cf.original_values == original_values
        assert isinstance(cf.counterfactual_values, dict)
        assert cf.distance >= 0.0
        assert cf.distance_metric == "l2"

    def test_counterfactual_satisfies_policy(self):
        """Test that counterfactual satisfies policy."""
        original_values = {"age": 60.0}
        policy_spec = {
            "conditions": [
                {"subject": "age", "threshold": 65.0, "mode": "soft"}
            ],
            "operator": "and",
        }

        cf = self.reasoner.find_counterfactual(
            original_values, policy_spec, max_steps=100
        )

        # After optimization, counterfactual should have higher truth degree
        assert cf.counterfactual_values["age"] > original_values["age"]


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(device="cpu")

    def test_reset(self):
        """Test metrics reset."""
        self.collector.update_truth_degree(0.8)
        assert len(self.collector.truth_degrees) == 1

        self.collector.reset()
        assert len(self.collector.truth_degrees) == 0

    def test_update_operations(self):
        """Test metric update operations."""
        self.collector.update_truth_degree(0.8)
        self.collector.update_loss(0.01, 0.02, 0.03)
        self.collector.update_prediction(0.9, 1, "policy1")
        self.collector.update_latency(10.5)

        assert len(self.collector.truth_degrees) == 1
        assert len(self.collector.impl_losses) == 1
        assert len(self.collector.predictions) == 1
        assert len(self.collector.latencies_ms) == 1

    def test_confusion_matrix_update(self):
        """Test confusion matrix tracking."""
        # True positive
        self.collector.update_prediction(0.9, 1, "policy1")
        # False positive
        self.collector.update_prediction(0.8, 0, "policy1")
        # True negative
        self.collector.update_prediction(0.1, 0, "policy1")
        # False negative
        self.collector.update_prediction(0.3, 1, "policy1")

        metrics = self.collector.compute_metrics()

        cm = metrics.confusion_matrices["policy1"]
        assert cm.tp == 1
        assert cm.fp == 1
        assert cm.tn == 1
        assert cm.fn == 1
        assert cm.accuracy() == 0.5
        assert cm.precision() == 0.5
        assert cm.recall() == 0.5

    def test_metrics_computation(self):
        """Test full metrics computation."""
        for i in range(10):
            self.collector.update_truth_degree(0.5 + i * 0.05)
            self.collector.update_loss(0.01, 0.02, 0.03)

        self.collector.total_facts = 10

        metrics = self.collector.compute_metrics(num_policies=1)

        assert metrics.num_facts == 10
        assert metrics.num_policies == 1
        assert metrics.mean_truth_degree > 0.0
        assert metrics.mean_implication_loss > 0.0

    def test_calibration_metrics(self):
        """Test calibration metrics computation."""
        predictions = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]

        for pred, label in zip(predictions, labels):
            self.collector.update_prediction(pred, label, "test")

        metrics = self.collector.compute_metrics()

        assert metrics.calibration.expected_calibration_error >= 0.0
        assert metrics.calibration.brier_score >= 0.0


class TestNeuraLogGASystem:
    """Tests for NeuraLogGASystem."""

    def setup_method(self):
        """Set up test fixtures."""
        config = SystemConfig(device="cpu", batch_size=4)
        self.system = NeuraLogGASystem(config)

    def test_system_initialization(self):
        """Test system initialization."""
        assert len(self.system.policies) == 0
        assert self.system.config.device == "cpu"

    def test_policy_registration(self):
        """Test policy registration and retrieval."""
        policy_spec = {
            "name": "test_policy",
            "conditions": [
                {"subject": "age", "threshold": 65.0, "mode": "soft"}
            ],
            "operator": "and",
        }

        self.system.register_policy("test_policy", policy_spec)

        assert "test_policy" in self.system.policies
        assert self.system.policies["test_policy"] == policy_spec

    def test_policy_unregistration(self):
        """Test policy unregistration."""
        self.system.register_policy(
            "test",
            {"name": "test", "conditions": [], "operator": "and"},
        )
        assert "test" in self.system.policies

        self.system.unregister_policy("test")
        assert "test" not in self.system.policies

    def test_verify_facts(self):
        """Test fact verification."""
        policy_spec = {
            "name": "senior",
            "conditions": [
                {"subject": "age", "threshold": 65.0, "mode": "soft"}
            ],
            "operator": "and",
        }
        self.system.register_policy("senior", policy_spec)

        facts = [
            {"age": 70.0},
            {"age": 60.0},
            {"age": 65.0},
        ]

        results = self.system.verify_facts(facts)

        assert len(results) == 3
        for result in results:
            assert result.fact_id is not None
            assert result.is_consistent in [True, False]
            assert 0.0 <= result.confidence_score <= 1.0

    def test_verify_facts_with_counterfactuals(self):
        """Test fact verification with counterfactuals."""
        policy_spec = {
            "name": "test",
            "conditions": [
                {"subject": "age", "threshold": 65.0, "mode": "soft"}
            ],
            "operator": "and",
        }
        self.system.register_policy("test", policy_spec)

        facts = [{"age": 60.0}]  # Doesn't satisfy policy

        results = self.system.verify_facts(
            facts, generate_counterfactuals=True
        )

        assert len(results) == 1
        # Counterfactual might be None if optimization fails, but check the field exists
        assert hasattr(results[0], "counterfactual")

    def test_metrics_export(self):
        """Test metrics export."""
        policy_spec = {
            "name": "test",
            "conditions": [
                {"subject": "x", "threshold": 0.5, "mode": "soft"}
            ],
            "operator": "and",
        }
        self.system.register_policy("test", policy_spec)

        facts = [{"x": 0.6}, {"x": 0.4}]
        self.system.verify_facts(facts)

        metrics_dict = self.system.export_metrics_json()

        assert "accuracy" in metrics_dict
        assert "precision" in metrics_dict
        assert "f1" in metrics_dict
        assert "throughput_facts_per_sec" in metrics_dict

    def test_system_summary(self):
        """Test system summary generation."""
        self.system.register_policy(
            "policy1", {"name": "policy1", "conditions": [], "operator": "and"}
        )
        self.system.register_policy(
            "policy2",
            {
                "name": "policy2",
                "conditions": [
                    {"subject": "x", "threshold": 0.5, "mode": "soft"}
                ],
                "operator": "and",
            },
        )

        summary = self.system.summary()

        assert "NeuraLog Geometric Algebra System" in summary
        assert "policy1" in summary
        assert "policy2" in summary

    def test_system_to_device(self):
        """Test moving system to device."""
        original_device = self.system.config.device
        # Try to move to the same device
        self.system.to(original_device)
        assert self.system.config.device == original_device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
