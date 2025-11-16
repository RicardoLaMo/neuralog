"""Geometric Algebra-aware Reasoner for NeuraLog.

Integrates Geometric Algebra truth degrees and policy verification
with the existing knowledge graph and symbolic reasoning infrastructure.

This reasoner bridges:
1. KnowledgeGraph (Triple-based representation)
2. GeometricAlgebra (Cl(3,0) triplet logic)
3. Symbolic reasoning (OWL, SPARQL, Z3 verification)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralog.core.types import ConfidenceLevel, Entity, KnowledgeGraph, Relation, Triple
from neuralog.symbolic.geometric_algebra.policy import Policy, PolicyCondition, PolicyRule
from neuralog.symbolic.geometric_algebra.triplet_logic import (
    ThresholdMode,
    TripletLogicLayer,
)


class GATripleEvaluation:
    """Evaluation result for a triple in Geometric Algebra."""

    def __init__(
        self,
        triple: Triple,
        truth_degree: torch.Tensor,
        canonical_state: torch.Tensor,
        confidence_level: ConfidenceLevel,
    ):
        """Initialize evaluation result.

        Args:
            triple: Original knowledge graph triple
            truth_degree: Computed truth degree [0, 1]
            canonical_state: Canonical vector state [8]
            confidence_level: Mapped confidence level
        """
        self.triple = triple
        self.truth_degree = truth_degree
        self.canonical_state = canonical_state
        self.confidence_level = confidence_level

    def to_triple(self) -> Triple:
        """Convert evaluation back to Triple with updated confidence.

        Returns:
            Triple with updated confidence values from truth degree
        """
        updated_triple = Triple(
            subject=self.triple.subject,
            predicate=self.triple.predicate,
            object=self.triple.object,
            confidence=float(self.truth_degree.item()),
            confidence_level=self.confidence_level,
            provenance=self.triple.provenance or {},
            verification_proof=self.triple.verification_proof,
            timestamp=self.triple.timestamp,
            id=self.triple.id,
        )
        return updated_triple


class GeometricAlgebraReasoner(nn.Module):
    """Reasoner integrating Geometric Algebra with knowledge graphs.

    Evaluates triples and policies using GA truth degrees and integrates
    with the existing NeuraLog symbolic reasoning pipeline.

    Attributes:
        triplet_logic: Core TripletLogicLayer for GA operations
        policies: Registered policies for verification
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        beta: float = 20.0,
        margin: float = 0.5,
    ):
        """Initialize the GA Reasoner.

        Args:
            epsilon: Numerical stability constant
            beta: Sigmoid steepness parameter
            margin: Margin for MARGIN threshold mode
        """
        super().__init__()
        self.triplet_logic = TripletLogicLayer(
            epsilon=epsilon,
            beta=beta,
            margin=margin,
        )
        self.policies: Dict[str, Policy] = {}

    def register_policy(self, policy: Policy) -> None:
        """Register a policy for verification.

        Args:
            policy: Policy to register
        """
        self.policies[policy.name] = policy

    def evaluate_numeric_triple(
        self,
        subject: str,
        predicate: str,
        value: float,
        threshold: float,
        threshold_mode: ThresholdMode = ThresholdMode.SOFT,
    ) -> GATripleEvaluation:
        """Evaluate a numeric subject-predicate-value triple.

        Example:
            subject="age", predicate=">=", value=70, threshold=65

        Args:
            subject: Subject identifier
            predicate: Predicate description
            value: Numeric value
            threshold: Threshold for evaluation
            threshold_mode: How to evaluate threshold

        Returns:
            GATripleEvaluation with truth degree and confidence
        """
        value_tensor = torch.tensor(value, dtype=torch.float32)
        threshold_tensor = torch.tensor(threshold, dtype=torch.float32)

        # Create canonical state and compute truth degree
        canonical_state = self.triplet_logic.numeric_triplet_state(
            value_tensor, threshold_tensor, threshold_mode
        )
        truth_degree = self.triplet_logic.truth_degree(canonical_state)

        # Map truth degree to confidence level
        t = float(truth_degree.item())
        if t > 0.99:
            confidence_level = ConfidenceLevel.VERIFIED
        elif t > 0.90:
            confidence_level = ConfidenceLevel.HIGH
        elif t > 0.70:
            confidence_level = ConfidenceLevel.MEDIUM
        elif t > 0.0:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.UNCERTAIN

        # Create triple representation
        triple = Triple(
            subject=Entity(uri=f"concept:{subject}", label=subject),
            predicate=Relation(uri=f"relation:{predicate}", label=predicate),
            object=value,
            confidence=t,
            confidence_level=confidence_level,
        )

        return GATripleEvaluation(
            triple=triple,
            truth_degree=truth_degree,
            canonical_state=canonical_state,
            confidence_level=confidence_level,
        )

    def evaluate_knowledge_graph(
        self,
        kg: KnowledgeGraph,
        threshold_map: Optional[Dict[str, Tuple[float, ThresholdMode]]] = None,
    ) -> Dict[str, GATripleEvaluation]:
        """Evaluate all triples in a knowledge graph.

        Args:
            kg: Knowledge graph to evaluate
            threshold_map: Mapping of predicates to (threshold, mode) tuples.
                          If None, skips numeric evaluation.

        Returns:
            Dict mapping triple IDs to GATripleEvaluation results
        """
        results = {}

        for triple in kg.triples:
            try:
                # Try to extract numeric value from object
                if isinstance(triple.object, (int, float)):
                    value = float(triple.object)
                    predicate_str = (
                        triple.predicate.label or triple.predicate.uri
                        if isinstance(triple.predicate, Relation)
                        else str(triple.predicate)
                    )

                    # Check if we have threshold info for this predicate
                    if (
                        threshold_map
                        and predicate_str in threshold_map
                    ):
                        threshold, mode = threshold_map[predicate_str]
                        subject_str = (
                            triple.subject.label or triple.subject.uri
                            if isinstance(triple.subject, Entity)
                            else str(triple.subject)
                        )

                        evaluation = self.evaluate_numeric_triple(
                            subject=subject_str,
                            predicate=predicate_str,
                            value=value,
                            threshold=threshold,
                            threshold_mode=mode,
                        )
                        results[triple.id] = evaluation

            except (ValueError, TypeError):
                # Skip triples that can't be evaluated numerically
                pass

        return results

    def verify_policy_against_kg(
        self,
        policy_name: str,
        kg: KnowledgeGraph,
        entity_values: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Verify a policy against knowledge graph values.

        Args:
            policy_name: Name of registered policy
            kg: Knowledge graph for reference
            entity_values: Dict mapping entity names to numeric values

        Returns:
            Tuple of (total_loss, rule_results) where rule_results maps
            rule names to (antecedent_truth, consequent_truth) pairs

        Raises:
            ValueError: If policy not registered
        """
        if policy_name not in self.policies:
            raise ValueError(f"Policy '{policy_name}' not registered")

        policy = self.policies[policy_name]

        # Convert values to tensors
        tensor_values = {
            name: torch.tensor(val, dtype=torch.float32)
            for name, val in entity_values.items()
        }

        # Evaluate policy rules
        rule_results = policy.evaluate(tensor_values)

        # Compute total loss
        total_loss = policy.total_loss(tensor_values)

        return total_loss, rule_results

    def extract_truth_degrees_from_kg(
        self,
        kg: KnowledgeGraph,
        threshold_map: Dict[str, Tuple[float, ThresholdMode]],
    ) -> Dict[str, torch.Tensor]:
        """Extract truth degrees for all evaluable entities in KG.

        Args:
            kg: Knowledge graph
            threshold_map: Mapping of entity types to (threshold, mode)

        Returns:
            Dict mapping entity names to truth degrees
        """
        truth_degrees = {}

        evaluations = self.evaluate_knowledge_graph(kg, threshold_map)
        for eval_result in evaluations.values():
            subject_str = (
                eval_result.triple.subject.label
                or eval_result.triple.subject.uri
                if isinstance(eval_result.triple.subject, Entity)
                else str(eval_result.triple.subject)
            )
            truth_degrees[subject_str] = eval_result.truth_degree

        return truth_degrees

    def forward(
        self,
        kg: KnowledgeGraph,
        threshold_map: Optional[Dict[str, Tuple[float, ThresholdMode]]] = None,
    ) -> Dict[str, GATripleEvaluation]:
        """Forward pass: evaluate all triples in knowledge graph.

        Args:
            kg: Knowledge graph to evaluate
            threshold_map: Optional threshold mapping for numeric predicates

        Returns:
            Dict mapping triple IDs to evaluation results
        """
        return self.evaluate_knowledge_graph(kg, threshold_map)
