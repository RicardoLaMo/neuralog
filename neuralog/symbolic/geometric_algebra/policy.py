"""Policy representation and verification using Geometric Algebra.

Implements policies as rules over triplet states with implications,
equivalences, and LLM-based validation.

Example policy:
  IF age >= 65 AND season = "low" AND budget >= 22
  THEN eligible_for_discount = TRUE
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralog.symbolic.geometric_algebra.triplet_logic import (
    ThresholdMode,
    TripletLogicLayer,
)


class PolicyCondition(nn.Module):
    """Single condition in a policy (subject-predicate-object triplet).

    Example: age >= 65
    """

    def __init__(
        self,
        subject: str,
        predicate: str,
        threshold: Optional[float] = None,
        threshold_mode: ThresholdMode = ThresholdMode.SOFT,
    ):
        """Initialize policy condition.

        Args:
            subject: Subject identifier (e.g., "age", "budget")
            predicate: Predicate description (e.g., ">=", "=", "in")
            threshold: Numeric threshold (for numeric predicates)
            threshold_mode: How to evaluate threshold. Defaults to SOFT.
        """
        super().__init__()
        self.subject = subject
        self.predicate = predicate
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.triplet_logic = TripletLogicLayer()

    def evaluate(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate condition for given value.

        Args:
            value: Value to check against condition

        Returns:
            Tuple of (canonical_state, truth_degree)
        """
        if self.threshold is None:
            raise ValueError(f"Condition {self.subject} requires a threshold")

        threshold_tensor = torch.tensor(
            self.threshold, dtype=value.dtype, device=value.device
        )
        state = self.triplet_logic.numeric_triplet_state(
            value, threshold_tensor, self.threshold_mode
        )
        truth = self.triplet_logic.truth_degree(state)
        return state, truth


class PolicyRule(nn.Module):
    """Logical rule combining conditions with AND/OR operations.

    Example: condition1 AND condition2 AND condition3 IMPLIES conclusion
    """

    def __init__(
        self,
        name: str,
        conditions: List[PolicyCondition],
        conclusion: PolicyCondition,
        operator: str = "and",
    ):
        """Initialize policy rule.

        Args:
            name: Rule identifier
            conditions: List of conditions in antecedent
            conclusion: Single condition in consequent
            operator: How to combine conditions: "and" or "or"
        """
        super().__init__()
        self.name = name
        self.conditions = nn.ModuleList(conditions)
        self.conclusion = conclusion
        self.operator = operator
        self.triplet_logic = TripletLogicLayer()

        if operator not in ["and", "or"]:
            raise ValueError(f"Unknown operator: {operator}")

    def evaluate(
        self,
        condition_values: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate rule for given condition values.

        Args:
            condition_values: Dict mapping subject names to values

        Returns:
            Tuple of (antecedent_truth, consequent_truth)

        Raises:
            ValueError: If required condition values are missing
        """
        # Evaluate all conditions
        truth_degrees = []
        for condition in self.conditions:
            if condition.subject not in condition_values:
                raise ValueError(
                    f"Missing value for condition subject: {condition.subject}"
                )
            value = condition_values[condition.subject]
            _, truth = condition.evaluate(value)
            truth_degrees.append(truth)

        # Combine with specified operator
        if self.operator == "and":
            antecedent_truth = self.triplet_logic.t_and(*truth_degrees)
        else:  # "or"
            antecedent_truth = truth_degrees[0]
            for t in truth_degrees[1:]:
                antecedent_truth = self.triplet_logic.t_or(antecedent_truth, t)

        # Evaluate conclusion
        if self.conclusion.subject not in condition_values:
            raise ValueError(
                f"Missing value for conclusion subject: {self.conclusion.subject}"
            )
        conclusion_value = condition_values[self.conclusion.subject]
        _, consequent_truth = self.conclusion.evaluate(conclusion_value)

        return antecedent_truth, consequent_truth

    def implication_loss(
        self,
        condition_values: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute implication loss for rule: antecedent => consequent.

        Args:
            condition_values: Dict mapping subject names to values

        Returns:
            Loss value
        """
        antecedent, consequent = self.evaluate(condition_values)
        return self.triplet_logic.loss_implies(antecedent, consequent)


class Policy(nn.Module):
    """Complete policy consisting of multiple rules.

    Policies define decision-making logic with verifiable conditions
    and support both hard and soft constraint enforcement.
    """

    def __init__(self, name: str, rules: List[PolicyRule]):
        """Initialize policy.

        Args:
            name: Policy identifier
            rules: List of policy rules
        """
        super().__init__()
        self.name = name
        self.rules = nn.ModuleList(rules)
        self.triplet_logic = TripletLogicLayer()

    def evaluate(
        self,
        condition_values: Dict[str, torch.Tensor],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Evaluate all rules in policy.

        Args:
            condition_values: Dict mapping subject names to values

        Returns:
            Dict mapping rule names to (antecedent_truth, consequent_truth)
        """
        results = {}
        for rule in self.rules:
            antecedent, consequent = rule.evaluate(condition_values)
            results[rule.name] = (antecedent, consequent)
        return results

    def total_loss(
        self,
        condition_values: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Compute total implication loss across all rules.

        Args:
            condition_values: Dict mapping subject names to values
            weights: Optional weights for each rule. Defaults to uniform.

        Returns:
            Weighted sum of implication losses
        """
        if weights is None:
            weights = {rule.name: 1.0 / len(self.rules) for rule in self.rules}

        total = torch.tensor(0.0, device=list(condition_values.values())[0].device)
        for rule in self.rules:
            loss = rule.implication_loss(condition_values)
            weight = weights.get(rule.name, 1.0 / len(self.rules))
            total = total + weight * loss

        return total

    def verify_llm_extraction(
        self,
        llm_values: Dict[str, torch.Tensor],
        policy_values: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Verify LLM-extracted values against policy-derived truth.

        Computes bidirectional loss separating false positives and
        false negatives for answer validation.

        Args:
            llm_values: Values extracted by LLM
            policy_values: Values computed by policy

        Returns:
            Tuple of (fp_loss, fn_loss, detailed_losses)
            where detailed_losses maps subjects to (fp, fn) pairs
        """
        detailed_losses = {}
        total_fp = torch.tensor(0.0, device=list(llm_values.values())[0].device)
        total_fn = torch.tensor(0.0, device=list(llm_values.values())[0].device)

        for subject in llm_values:
            if subject in policy_values:
                t_llm = llm_values[subject]
                t_policy = policy_values[subject]
                fp, fn = self.triplet_logic.loss_bidirectional(t_llm, t_policy)
                detailed_losses[subject] = (fp, fn)
                total_fp = total_fp + fp
                total_fn = total_fn + fn

        return total_fp, total_fn, detailed_losses
