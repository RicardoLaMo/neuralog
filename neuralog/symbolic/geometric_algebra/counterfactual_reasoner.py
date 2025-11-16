"""Counterfactual reasoning with gradient-based policy refinement.

Finds minimal value perturbations needed to satisfy policies by solving
optimization problems via PyTorch autograd.

Key Features:
- Gradient-based search for policy-satisfying value combinations
- Rotor optimization for future GA-based transformations
- Distance metrics: L1 (Manhattan), L2 (Euclidean), L∞ (Chebyshev)
- Constraint satisfaction via Lagrange multipliers
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation for policy satisfaction."""

    original_values: Dict[str, float]
    counterfactual_values: Dict[str, float]
    distance: float
    distance_metric: str  # "l1", "l2", "linf"
    steps_to_satisfy: int
    final_truth_degree: float
    satisfied: bool
    constraints: List[str]  # Which conditions become satisfied


class CounterfactualReasoner(torch.nn.Module):
    """Gradient-based counterfactual reasoning for policies.

    Finds minimal perturbations to values that would satisfy a policy.
    Uses PyTorch optimization with support for:
    - Multiple distance metrics (L1, L2, L∞)
    - Constraint satisfaction via penalty methods
    - GPU acceleration for batch optimization
    - Differentiable loss landscapes

    Attributes:
        lr: Learning rate for optimization
        max_steps: Maximum optimization steps
        tolerance: Convergence tolerance
        distance_metric: Default distance metric
    """

    def __init__(
        self,
        lr: float = 0.1,
        max_steps: int = 100,
        tolerance: float = 1e-4,
        distance_metric: str = "l2",
        device: Optional[str] = None,
    ):
        """Initialize CounterfactualReasoner.

        Args:
            lr: Learning rate. Defaults to 0.1.
            max_steps: Max optimization steps. Defaults to 100.
            tolerance: Convergence tolerance. Defaults to 1e-4.
            distance_metric: "l1", "l2", or "linf". Defaults to "l2".
            device: PyTorch device. Auto-detects if None.
        """
        super().__init__()
        self.lr = lr
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.distance_metric = distance_metric

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

    def _compute_distance(
        self,
        original: torch.Tensor,
        counterfactual: torch.Tensor,
        metric: str = None,
    ) -> torch.Tensor:
        """Compute distance between original and counterfactual values.

        Args:
            original: Original values
            counterfactual: Counterfactual values
            metric: Distance metric. Uses default if None.

        Returns:
            Distance scalar
        """
        if metric is None:
            metric = self.distance_metric

        diff = counterfactual - original

        if metric == "l1":
            return torch.abs(diff).sum()
        elif metric == "l2":
            return torch.norm(diff, p=2)
        elif metric == "linf":
            return torch.abs(diff).max()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _evaluate_policy_satisfaction(
        self,
        values: torch.Tensor,
        policy_spec: Dict,
        epsilon: float = 1e-8,
        beta: float = 20.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate if values satisfy a policy (differentiable).

        Args:
            values: Policy parameter values
            policy_spec: Policy specification with conditions
            epsilon: Numerical stability
            beta: Sigmoid steepness

        Returns:
            Tuple of (truth_degree, loss) - both differentiable
        """
        conditions = policy_spec.get("conditions", [])
        operator = policy_spec.get("operator", "and")

        condition_truths = []

        for condition in conditions:
            value = values[condition["subject"]]
            threshold = condition["threshold"]
            mode = condition.get("mode", "soft")

            # Evaluate condition
            if mode == "hard":
                truth = (value >= threshold).float()
            elif mode == "soft":
                truth = torch.sigmoid(beta * (value - threshold))
            elif mode == "margin":
                margin = condition.get("margin", 0.5)
                truth = torch.sigmoid(beta * (value - threshold + margin))
            else:
                truth = torch.tensor(0.0, device=value.device)

            condition_truths.append(truth)

        # Combine conditions
        if operator == "and" and condition_truths:
            truth_degree = torch.stack(condition_truths).min()
        elif operator == "or" and condition_truths:
            truth_degree = condition_truths[0]
            for t in condition_truths[1:]:
                truth_degree = truth_degree + t - truth_degree * t
        else:
            truth_degree = torch.tensor(1.0, device=values.device)

        # Loss: penalize not satisfying policy
        loss = torch.clamp(
            1.0 - truth_degree, min=0.0
        ) ** 2  # Squared hinge loss

        return truth_degree, loss

    def find_counterfactual(
        self,
        original_values: Dict[str, float],
        policy_spec: Dict,
        value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        distance_metric: str = "l2",
    ) -> CounterfactualExplanation:
        """Find minimal perturbation satisfying a policy.

        Args:
            original_values: Original fact values
            policy_spec: Policy specification
            value_ranges: Optional (min, max) constraints per subject
            distance_metric: Distance metric to optimize

        Returns:
            CounterfactualExplanation with optimized values
        """
        # Convert to tensors
        subjects = list(original_values.keys())
        original_tensor = torch.tensor(
            [original_values[s] for s in subjects],
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        # Initialize counterfactual as copy of original
        counterfactual_var = torch.tensor(
            original_tensor.clone().detach(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        # Optimizer
        optimizer = optim.Adam([counterfactual_var], lr=self.lr)

        best_loss = float("inf")
        best_values = original_tensor.clone()
        converged_step = 0

        for step in range(self.max_steps):
            optimizer.zero_grad()

            # Create current value dict
            current_dict = {
                subjects[i]: counterfactual_var[i]
                for i in range(len(subjects))
            }

            # Evaluate policy satisfaction
            truth_degree, policy_loss = self._evaluate_policy_satisfaction(
                current_dict, policy_spec
            )

            # Distance loss (minimize perturbation)
            distance = self._compute_distance(
                original_tensor, counterfactual_var, distance_metric
            )

            # Combined loss: minimize distance while satisfying policy
            total_loss = policy_loss + 0.1 * distance  # Weight distance

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Apply value range constraints if provided
            if value_ranges is not None:
                with torch.no_grad():
                    for i, subject in enumerate(subjects):
                        if subject in value_ranges:
                            min_val, max_val = value_ranges[subject]
                            counterfactual_var[i].clamp_(min_val, max_val)

            # Track convergence
            loss_item = total_loss.item()
            if loss_item < best_loss:
                best_loss = loss_item
                best_values = counterfactual_var.clone().detach()
                converged_step = step

            # Early stopping if satisfied
            if truth_degree.item() > 0.95 and loss_item < self.tolerance:
                break

        # Extract results
        final_dict = {
            subjects[i]: best_values[i].item() for i in range(len(subjects))
        }
        final_distance = self._compute_distance(
            original_tensor, best_values, distance_metric
        ).item()

        # Re-evaluate with final values
        final_truth, final_loss = self._evaluate_policy_satisfaction(
            {subjects[i]: best_values[i] for i in range(len(subjects))},
            policy_spec,
        )
        satisfied = final_truth.item() > 0.5

        # Determine which constraints became satisfied
        constraints_satisfied = []
        conditions = policy_spec.get("conditions", [])
        for condition in conditions:
            subject = condition["subject"]
            threshold = condition["threshold"]
            if final_dict.get(subject, 0) >= threshold:
                constraints_satisfied.append(
                    f"{subject} >= {threshold}"
                )

        return CounterfactualExplanation(
            original_values=original_values,
            counterfactual_values=final_dict,
            distance=final_distance,
            distance_metric=distance_metric,
            steps_to_satisfy=converged_step,
            final_truth_degree=final_truth.item(),
            satisfied=satisfied,
            constraints=constraints_satisfied,
        )

    def batch_counterfactuals(
        self,
        batch_values: List[Dict[str, float]],
        policy_spec: Dict,
        value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[CounterfactualExplanation]:
        """Find counterfactuals for a batch of facts.

        Args:
            batch_values: List of fact value dicts
            policy_spec: Policy specification
            value_ranges: Optional constraints

        Returns:
            List of CounterfactualExplanation objects
        """
        results = []
        for fact_values in batch_values:
            cf = self.find_counterfactual(
                fact_values, policy_spec, value_ranges, self.distance_metric
            )
            results.append(cf)
        return results

    def rotor_optimization(
        self,
        initial_rotor: Optional[torch.Tensor] = None,
        target_vector: torch.Tensor = None,
        steps: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize rotor in Cl(3,0) to achieve target transformation.

        Placeholder for future GA-based rotor optimization.
        This will integrate with torch-ga once geometric product is implemented.

        Args:
            initial_rotor: Starting rotor [8,] (multivector)
            target_vector: Desired transformed vector [3,]
            steps: Optimization steps

        Returns:
            Tuple of (optimized_rotor, loss_history)
        """
        raise NotImplementedError(
            "Rotor optimization requires full geometric product. "
            "Implement after torch-ga integration."
        )

    def forward(
        self,
        original_values: Dict[str, float],
        policy_spec: Dict,
        value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> CounterfactualExplanation:
        """Forward pass: find counterfactual for policy satisfaction.

        Args:
            original_values: Original fact values
            policy_spec: Policy specification
            value_ranges: Optional constraints

        Returns:
            CounterfactualExplanation
        """
        return self.find_counterfactual(
            original_values, policy_spec, value_ranges, self.distance_metric
        )
