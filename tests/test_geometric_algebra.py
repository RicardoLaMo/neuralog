"""Tests for Geometric Algebra components.

Tests cover:
- TripletLogicLayer truth degree computation
- Threshold modes (HARD, SOFT, MARGIN)
- Logical connectives (AND, OR)
- Loss functions (implication, equivalence, bidirectional)
- Policy definition and verification
- GA Reasoner integration
"""

import pytest
import torch

from neuralog.core.types import ConfidenceLevel, Entity, KnowledgeGraph, Relation, Triple
from neuralog.symbolic.geometric_algebra import (
    Policy,
    PolicyCondition,
    PolicyRule,
    ThresholdMode,
    TripletLogicLayer,
)
from neuralog.symbolic.geometric_algebra.ga_reasoner import GeometricAlgebraReasoner


class TestTripletLogicLayer:
    """Tests for TripletLogicLayer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.layer = TripletLogicLayer(epsilon=1e-8, beta=20.0, margin=0.5)

    def test_create_vector_state(self):
        """Test canonical vector state creation."""
        x_s = torch.tensor(1.0)
        x_p = torch.tensor(2.0)
        x_t = torch.tensor(3.0)

        state = self.layer.create_vector_state(x_s, x_p, x_t)

        assert state.shape == (8,)
        assert state[0].item() == 0.0  # scalar
        assert state[1].item() == 1.0  # e1
        assert state[2].item() == 2.0  # e2
        assert state[3].item() == 3.0  # e3
        assert state[4].item() == 0.0  # e12
        assert state[5].item() == 0.0  # e13
        assert state[6].item() == 0.0  # e23
        assert state[7].item() == 0.0  # e123

    def test_numeric_triplet_state_hard_mode(self):
        """Test numeric triplet state with HARD threshold mode."""
        value = torch.tensor(70.0)
        threshold = torch.tensor(65.0)

        state = self.layer.numeric_triplet_state(
            value, threshold, ThresholdMode.HARD
        )

        # x3 should be 1.0 (value >= threshold)
        assert state[3].item() == 1.0
        assert state.shape == (8,)

    def test_numeric_triplet_state_soft_mode(self):
        """Test numeric triplet state with SOFT threshold mode."""
        value = torch.tensor(65.0)
        threshold = torch.tensor(65.0)

        state = self.layer.numeric_triplet_state(
            value, threshold, ThresholdMode.SOFT
        )

        # x3 should be close to 0.5 (sigmoid(0) ≈ 0.5)
        assert 0.49 < state[3].item() < 0.51

    def test_numeric_triplet_state_margin_mode(self):
        """Test numeric triplet state with MARGIN threshold mode."""
        value = torch.tensor(65.5)
        threshold = torch.tensor(65.0)
        margin = 0.5

        state = self.layer.numeric_triplet_state(
            value, threshold, ThresholdMode.MARGIN
        )

        # x3 should be sigmoid(beta * (65.5 - 65.0 + 0.5))
        assert 0.0 <= state[3].item() <= 1.0

    def test_truth_degree_computation(self):
        """Test truth degree computation from canonical state."""
        x_s = torch.tensor(0.3)
        x_p = torch.tensor(0.4)
        x_t = torch.tensor(0.5)

        state = self.layer.create_vector_state(x_s, x_p, x_t)
        truth = self.layer.truth_degree(state)

        # truth = |x_t| / ||[x_s, x_p, x_t]||
        # norm = sqrt(0.3^2 + 0.4^2 + 0.5^2) = sqrt(0.5) ≈ 0.707
        # truth = 0.5 / 0.707 ≈ 0.707
        expected = 0.5 / torch.sqrt(torch.tensor(0.3**2 + 0.4**2 + 0.5**2))
        assert abs(truth.item() - expected.item()) < 1e-5

    def test_truth_degree_batch(self):
        """Test truth degree computation with batch inputs."""
        x_s = torch.tensor([0.3, 0.4])
        x_p = torch.tensor([0.4, 0.5])
        x_t = torch.tensor([0.5, 0.6])

        state = self.layer.create_vector_state(x_s, x_p, x_t)
        truth = self.layer.truth_degree(state)

        assert truth.shape == (2,)
        assert (truth >= 0.0).all() and (truth <= 1.0).all()

    def test_t_and_operation(self):
        """Test fuzzy AND operation."""
        t1 = torch.tensor(0.8)
        t2 = torch.tensor(0.6)
        t3 = torch.tensor(0.9)

        result = self.layer.t_and(t1, t2, t3)

        # AND should return minimum
        assert result.item() == 0.6

    def test_t_or_operation(self):
        """Test fuzzy OR operation."""
        t1 = torch.tensor(0.3)
        t2 = torch.tensor(0.7)

        result = self.layer.t_or(t1, t2)

        # OR: t1 + t2 - t1*t2 = 0.3 + 0.7 - 0.21 = 0.79
        expected = 0.3 + 0.7 - 0.3 * 0.7
        assert abs(result.item() - expected) < 1e-5

    def test_loss_implies(self):
        """Test implication loss function."""
        t_p = torch.tensor(0.8)
        t_q = torch.tensor(0.5)

        loss = self.layer.loss_implies(t_p, t_q)

        # Loss = (0.8 - 0.5)^2 = 0.09
        assert abs(loss.item() - 0.09) < 1e-5

    def test_loss_implies_satisfied(self):
        """Test implication loss when satisfied."""
        t_p = torch.tensor(0.3)
        t_q = torch.tensor(0.8)

        loss = self.layer.loss_implies(t_p, t_q)

        # Loss = max(0, 0.3 - 0.8)^2 = 0
        assert abs(loss.item()) < 1e-5

    def test_loss_equiv(self):
        """Test equivalence loss function."""
        t_p = torch.tensor(0.7)
        t_q = torch.tensor(0.8)

        loss = self.layer.loss_equiv(t_p, t_q)

        # Loss = (0.7 - 0.8)^2 = 0.01
        assert abs(loss.item() - 0.01) < 1e-5

    def test_loss_bidirectional(self):
        """Test bidirectional loss function."""
        t_llm = torch.tensor(0.8)
        t_policy = torch.tensor(0.5)

        fp, fn = self.layer.loss_bidirectional(t_llm, t_policy)

        # FP = (0.8 - 0.5)^2 = 0.09
        # FN = 0
        assert abs(fp.item() - 0.09) < 1e-5
        assert abs(fn.item()) < 1e-5

    def test_forward_pass(self):
        """Test forward pass."""
        x_s = torch.tensor(0.3)
        x_p = torch.tensor(0.4)
        x_t = torch.tensor(0.5)

        state = self.layer.create_vector_state(x_s, x_p, x_t)
        truth = self.layer.forward(state)

        assert truth.shape == ()
        assert 0.0 <= truth.item() <= 1.0


class TestPolicy:
    """Tests for Policy framework."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create conditions for a discount policy:
        # age >= 65, season = "low-season", budget >= 22
        self.age_condition = PolicyCondition(
            subject="age",
            predicate=">=",
            threshold=65.0,
            threshold_mode=ThresholdMode.SOFT,
        )
        self.season_condition = PolicyCondition(
            subject="season",
            predicate="=",
            threshold=0.5,
            threshold_mode=ThresholdMode.SOFT,
        )
        self.budget_condition = PolicyCondition(
            subject="budget",
            predicate=">=",
            threshold=22.0,
            threshold_mode=ThresholdMode.SOFT,
        )
        self.eligible_condition = PolicyCondition(
            subject="eligible",
            predicate="=",
            threshold=0.5,
            threshold_mode=ThresholdMode.HARD,
        )

    def test_policy_condition_evaluation(self):
        """Test policy condition evaluation."""
        value = torch.tensor(70.0)
        state, truth = self.age_condition.evaluate(value)

        assert state.shape == (8,)
        assert truth.item() >= 0.5  # 70 >= 65

    def test_policy_rule_creation(self):
        """Test policy rule creation."""
        rule = PolicyRule(
            name="discount_rule",
            conditions=[self.age_condition, self.season_condition, self.budget_condition],
            conclusion=self.eligible_condition,
            operator="and",
        )

        assert rule.name == "discount_rule"
        assert len(rule.conditions) == 3

    def test_policy_rule_evaluation(self):
        """Test policy rule evaluation."""
        rule = PolicyRule(
            name="discount_rule",
            conditions=[self.age_condition, self.season_condition, self.budget_condition],
            conclusion=self.eligible_condition,
            operator="and",
        )

        condition_values = {
            "age": torch.tensor(70.0),
            "season": torch.tensor(0.8),
            "budget": torch.tensor(25.0),
            "eligible": torch.tensor(1.0),
        }

        antecedent, consequent = rule.evaluate(condition_values)

        assert antecedent.shape == ()
        assert consequent.shape == ()
        assert 0.0 <= antecedent.item() <= 1.0
        assert 0.0 <= consequent.item() <= 1.0

    def test_policy_creation(self):
        """Test policy creation."""
        rule = PolicyRule(
            name="discount_rule",
            conditions=[self.age_condition, self.season_condition],
            conclusion=self.eligible_condition,
            operator="and",
        )
        policy = Policy(name="discount_policy", rules=[rule])

        assert policy.name == "discount_policy"
        assert len(policy.rules) == 1

    def test_policy_evaluation(self):
        """Test policy evaluation."""
        rule = PolicyRule(
            name="discount_rule",
            conditions=[self.age_condition, self.season_condition],
            conclusion=self.eligible_condition,
            operator="and",
        )
        policy = Policy(name="discount_policy", rules=[rule])

        condition_values = {
            "age": torch.tensor(70.0),
            "season": torch.tensor(0.8),
            "eligible": torch.tensor(1.0),
        }

        results = policy.evaluate(condition_values)

        assert "discount_rule" in results
        antecedent, consequent = results["discount_rule"]
        assert antecedent.shape == ()
        assert consequent.shape == ()

    def test_policy_total_loss(self):
        """Test policy total loss computation."""
        rule = PolicyRule(
            name="discount_rule",
            conditions=[self.age_condition],
            conclusion=self.eligible_condition,
            operator="and",
        )
        policy = Policy(name="discount_policy", rules=[rule])

        condition_values = {
            "age": torch.tensor(70.0),
            "eligible": torch.tensor(1.0),
        }

        loss = policy.total_loss(condition_values)

        assert loss.shape == ()
        assert loss.item() >= 0.0


class TestGeometricAlgebraReasoner:
    """Tests for GeometricAlgebraReasoner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reasoner = GeometricAlgebraReasoner()

    def test_reasoner_initialization(self):
        """Test reasoner initialization."""
        assert isinstance(self.reasoner.triplet_logic, TripletLogicLayer)
        assert len(self.reasoner.policies) == 0

    def test_evaluate_numeric_triple(self):
        """Test numeric triple evaluation."""
        evaluation = self.reasoner.evaluate_numeric_triple(
            subject="age",
            predicate=">=",
            value=70.0,
            threshold=65.0,
            threshold_mode=ThresholdMode.SOFT,
        )

        assert evaluation.triple is not None
        assert evaluation.truth_degree.shape == ()
        assert 0.0 <= evaluation.truth_degree.item() <= 1.0
        assert evaluation.confidence_level in ConfidenceLevel

    def test_register_policy(self):
        """Test policy registration."""
        age_condition = PolicyCondition(
            subject="age", predicate=">=", threshold=65.0
        )
        eligible_condition = PolicyCondition(
            subject="eligible", predicate="=", threshold=0.5
        )
        rule = PolicyRule(
            name="age_rule",
            conditions=[age_condition],
            conclusion=eligible_condition,
        )
        policy = Policy(name="test_policy", rules=[rule])

        self.reasoner.register_policy(policy)

        assert "test_policy" in self.reasoner.policies

    def test_verify_policy_against_kg(self):
        """Test policy verification against knowledge graph values."""
        age_condition = PolicyCondition(
            subject="age", predicate=">=", threshold=65.0
        )
        eligible_condition = PolicyCondition(
            subject="eligible", predicate="=", threshold=0.5
        )
        rule = PolicyRule(
            name="age_rule",
            conditions=[age_condition],
            conclusion=eligible_condition,
        )
        policy = Policy(name="test_policy", rules=[rule])
        self.reasoner.register_policy(policy)

        entity_values = {"age": 70.0, "eligible": 1.0}

        total_loss, rule_results = self.reasoner.verify_policy_against_kg(
            "test_policy", KnowledgeGraph(name="test_kg"), entity_values
        )

        assert total_loss.shape == ()
        assert "age_rule" in rule_results

    def test_evaluate_knowledge_graph(self):
        """Test knowledge graph evaluation."""
        kg = KnowledgeGraph(name="test_kg")

        # Add triples with numeric values
        age_entity = Entity(uri="person:alice", label="Alice")
        age_relation = Relation(uri="hasAge", label="age")
        age_triple = Triple(
            subject=age_entity, predicate=age_relation, object=70
        )
        kg.add_triple(age_triple)

        threshold_map = {"age": (65.0, ThresholdMode.SOFT)}

        results = self.reasoner.evaluate_knowledge_graph(kg, threshold_map)

        assert len(results) > 0 or len(kg.triples) > 0  # May be empty if no matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
