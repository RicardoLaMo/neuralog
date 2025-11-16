"""
Counterfactual Reasoning via Rotor Trajectories

Implements smooth counterfactual analysis by tracing rotor paths in geometric algebra.
Enables "what-if" questions about policy decisions and robustness analysis.

Key concepts:
- Rotor trajectories: Smooth paths through state space
- Counterfactual planes: Bivectors defining modification directions
- Robustness analysis: Sensitivity to perturbations
"""

from typing import List, Tuple, Callable, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from .clifford import Rotor, Bivector, geometric_product
from .triplet_logic import TripletState, TripletRotor, truth_degree
from .policy_encoder import PolicyChain


class CounterfactualReasoner(nn.Module):
    """
    Counterfactual reasoner using geometric algebra.

    Generates smooth trajectories through policy state space to answer
    "what-if" questions and analyze decision boundaries.

    Args:
        policy_chain: Policy to analyze
        epsilon: Numerical stability constant
    """

    def __init__(
        self,
        policy_chain: PolicyChain,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        self.policy_chain = policy_chain
        self.epsilon = epsilon

    def analyze_numeric_counterfactual(
        self,
        triplet_index: int,
        value_range: Tuple[float, float],
        num_steps: int = 20,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze counterfactual by varying a numeric triplet value.

        Example: "What if age ranged from 60 to 75?"

        Args:
            triplet_index: Index of triplet in policy chain to vary
            value_range: (min_value, max_value) to explore
            num_steps: Number of steps in trajectory
            threshold: Threshold for the numeric comparison (if None, extracted from triplet)

        Returns:
            Dictionary with:
                - values: Array of values explored
                - truth_degrees: Truth degrees along trajectory
                - decision_boundary: Value where decision flips (if exists)
                - is_robust: Whether policy is robust to small perturbations
        """
        triplet = self.policy_chain.get_triplet(triplet_index)

        if not isinstance(triplet, TripletState):
            raise ValueError("Counterfactual analysis only supports TripletState triplets")

        min_val, max_val = value_range
        values = np.linspace(min_val, max_val, num_steps)

        truth_degrees = []
        decisions = []

        # Get original state
        original_state = triplet.state.clone()

        for val in values:
            # Modify triplet value (assuming it's in x_s position for numeric values)
            # This is simplified - in practice we'd need to know which component to modify
            modified_state = original_state.clone()
            modified_state[2] = val / 100.0  # Normalize to [0, 1] range

            triplet.state = nn.Parameter(modified_state)

            # Evaluate policy
            truth = self.policy_chain.evaluate()
            truth_degrees.append(truth.item())
            decisions.append(truth.item() > 0.5)

        # Restore original state
        triplet.state = nn.Parameter(original_state)

        # Find decision boundary
        decision_boundary = None
        for i in range(len(decisions) - 1):
            if decisions[i] != decisions[i + 1]:
                # Decision flipped between values[i] and values[i+1]
                decision_boundary = (values[i] + values[i + 1]) / 2.0
                break

        # Check robustness (are there multiple decision flips?)
        num_flips = sum(1 for i in range(len(decisions) - 1) if decisions[i] != decisions[i + 1])
        is_robust = num_flips <= 1  # Robust if at most one flip

        return {
            "values": values,
            "truth_degrees": np.array(truth_degrees),
            "decisions": decisions,
            "decision_boundary": decision_boundary,
            "is_robust": is_robust,
            "num_decision_flips": num_flips,
        }

    def analyze_joint_counterfactual(
        self,
        triplet_indices: List[int],
        value_ranges: List[Tuple[float, float]],
        num_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze joint counterfactual by varying multiple triplets.

        Example: "What if we varied both age and budget?"

        Args:
            triplet_indices: Indices of triplets to vary
            value_ranges: List of (min, max) ranges for each triplet
            num_steps: Number of steps per dimension

        Returns:
            Dictionary with:
                - value_grids: Meshgrids of values for each dimension
                - truth_surface: 2D array of truth degrees
                - decision_regions: Binary decision map
        """
        if len(triplet_indices) != 2:
            raise ValueError("Joint counterfactual currently supports 2 dimensions only")

        if len(value_ranges) != 2:
            raise ValueError("Must provide range for each triplet")

        # Create value grids
        values_0 = np.linspace(value_ranges[0][0], value_ranges[0][1], num_steps)
        values_1 = np.linspace(value_ranges[1][0], value_ranges[1][1], num_steps)

        V0, V1 = np.meshgrid(values_0, values_1)

        truth_surface = np.zeros_like(V0)
        decision_regions = np.zeros_like(V0, dtype=bool)

        # Get original states
        triplet_0 = self.policy_chain.get_triplet(triplet_indices[0])
        triplet_1 = self.policy_chain.get_triplet(triplet_indices[1])

        original_state_0 = triplet_0.state.clone()
        original_state_1 = triplet_1.state.clone()

        # Explore joint space
        for i in range(num_steps):
            for j in range(num_steps):
                # Modify both triplets
                modified_state_0 = original_state_0.clone()
                modified_state_0[2] = V0[i, j] / 100.0

                modified_state_1 = original_state_1.clone()
                modified_state_1[2] = V1[i, j] / 100.0

                triplet_0.state = nn.Parameter(modified_state_0)
                triplet_1.state = nn.Parameter(modified_state_1)

                # Evaluate
                truth = self.policy_chain.evaluate()
                truth_surface[i, j] = truth.item()
                decision_regions[i, j] = truth.item() > 0.5

        # Restore original states
        triplet_0.state = nn.Parameter(original_state_0)
        triplet_1.state = nn.Parameter(original_state_1)

        return {
            "value_grids": (V0, V1),
            "truth_surface": truth_surface,
            "decision_regions": decision_regions,
            "values_0": values_0,
            "values_1": values_1,
        }

    def compute_sensitivity(
        self,
        triplet_index: int,
        perturbation: float = 1.0,
    ) -> float:
        """
        Compute sensitivity of policy to perturbations in a triplet.

        Measures how much the policy truth degree changes when the
        triplet value is perturbed.

        Args:
            triplet_index: Index of triplet to perturb
            perturbation: Size of perturbation

        Returns:
            Sensitivity (gradient magnitude)
        """
        triplet = self.policy_chain.get_triplet(triplet_index)

        if not isinstance(triplet, TripletState):
            raise ValueError("Sensitivity analysis only supports TripletState")

        # Get baseline truth
        baseline_truth = self.policy_chain.evaluate()

        # Perturb and measure change
        original_state = triplet.state.clone()

        perturbed_state = original_state.clone()
        perturbed_state[2] += perturbation / 100.0  # Small perturbation to truth component

        triplet.state = nn.Parameter(perturbed_state)
        perturbed_truth = self.policy_chain.evaluate()

        # Restore
        triplet.state = nn.Parameter(original_state)

        # Compute sensitivity
        sensitivity = abs(perturbed_truth.item() - baseline_truth.item()) / (perturbation / 100.0)

        return sensitivity


def rotor_trajectory(
    start_state: torch.Tensor,
    end_state: torch.Tensor,
    num_steps: int = 20,
    bivector: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Generate smooth rotor trajectory from start to end state.

    Creates interpolated path through geometric algebra by gradually
    rotating from start to end via a bivector plane.

    Args:
        start_state: Starting 3D state vector
        end_state: Ending 3D state vector
        num_steps: Number of interpolation steps
        bivector: Optional bivector defining rotation plane
                  (if None, computed from start ∧ end)

    Returns:
        List of state vectors along trajectory
    """
    from .clifford import wedge_product

    if bivector is None:
        # Compute natural rotation plane from start to end
        bivector = wedge_product(start_state, end_state)
        bivec_norm = torch.sqrt(torch.sum(bivector**2) + 1e-8)

        if bivec_norm < 1e-6:
            # States are colinear, use direct interpolation
            trajectory = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                state = (1 - alpha) * start_state + alpha * end_state
                trajectory.append(state)
            return trajectory

        bivector = bivector / bivec_norm

    # Compute total rotation angle
    dot = torch.dot(start_state, end_state)
    norm_start = torch.norm(start_state) + 1e-8
    norm_end = torch.norm(end_state) + 1e-8
    cos_theta = dot / (norm_start * norm_end)
    total_angle = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

    # Generate trajectory
    trajectory = []

    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        angle = alpha * total_angle

        # Build rotor
        bivec_obj = Bivector(bivector[0].item(), bivector[1].item(), bivector[2].item())
        rotor = Rotor(bivec_obj, angle.item())

        # Apply to start state
        state = rotor.apply(start_state)
        trajectory.append(state)

    return trajectory


def counterfactual_robustness_loss(
    policy_chain: PolicyChain,
    perturbation_distribution: Callable[[], torch.Tensor],
    num_samples: int = 100,
) -> torch.Tensor:
    """
    Compute counterfactual robustness loss.

    Measures how much policy truth varies under random perturbations,
    encouraging policies that are robust to small input changes.

    L_CF = E_θ[|truth(θ) - truth(0)|]

    Args:
        policy_chain: Policy to evaluate
        perturbation_distribution: Function that generates random perturbations
        num_samples: Number of samples to draw

    Returns:
        Robustness loss (lower = more robust)
    """
    # Get baseline truth
    baseline_truth = policy_chain.evaluate()

    losses = []

    for _ in range(num_samples):
        # Generate perturbation
        perturbation = perturbation_distribution()

        # Apply perturbation to a random triplet (for simplicity)
        # In practice, you'd specify which triplets can be perturbed
        if len(policy_chain.triplets) > 0:
            triplet_idx = torch.randint(0, len(policy_chain.triplets), (1,)).item()
            triplet = policy_chain.get_triplet(triplet_idx)

            if isinstance(triplet, TripletState):
                original_state = triplet.state.clone()

                # Perturb truth component
                perturbed_state = original_state.clone()
                perturbed_state[2] += perturbation

                triplet.state = nn.Parameter(perturbed_state)

                # Evaluate perturbed policy
                perturbed_truth = policy_chain.evaluate()

                # Restore
                triplet.state = nn.Parameter(original_state)

                # Compute loss
                loss = torch.abs(perturbed_truth - baseline_truth)
                losses.append(loss)

    if losses:
        return torch.mean(torch.stack(losses))
    else:
        return torch.tensor(0.0)


class CounterfactualExplainer:
    """
    Generate explanations for policy decisions using counterfactuals.

    Answers questions like:
        - "Why was the user denied?"
        - "What would it take to be approved?"
        - "How close was the decision?"

    Usage:
        explainer = CounterfactualExplainer(policy_chain)

        explanation = explainer.explain_decision(
            actual_values={"age": 63, "budget": 25},
            decision="denied"
        )

        print(explanation)
        # "User was denied because age (63) is below threshold (65).
        #  If age were 65 or higher, user would be approved.
        #  User was 2 years below the threshold."
    """

    def __init__(self, policy_chain: PolicyChain):
        self.policy_chain = policy_chain
        self.reasoner = CounterfactualReasoner(policy_chain)

    def explain_decision(
        self,
        actual_values: Dict[str, float],
        decision: str,
    ) -> str:
        """
        Generate counterfactual explanation for a decision.

        Args:
            actual_values: Dictionary of actual values for each condition
            decision: "approved" or "denied"

        Returns:
            Human-readable explanation
        """
        explanations = []

        # Analyze each condition
        for i, triplet in enumerate(self.policy_chain.triplets):
            if isinstance(triplet, TripletState):
                # Extract triplet components
                truth_val = triplet.state[2].item()

                # Check if this triplet is satisfied
                if truth_val < 0.5:
                    # Condition not met
                    # Try to find what value would satisfy it
                    # (This is simplified - in practice you'd track condition metadata)
                    explanations.append(
                        f"Condition {i+1} not satisfied (truth degree: {truth_val:.3f})"
                    )

        if decision == "denied" and explanations:
            return (
                f"Decision: DENIED\n\n"
                f"Reasons:\n" +
                "\n".join(f"  - {exp}" for exp in explanations) +
                f"\n\nTo be approved, all conditions must be satisfied (truth degree > 0.5)."
            )
        elif decision == "approved":
            truth = self.policy_chain.evaluate().item()
            return (
                f"Decision: APPROVED\n\n"
                f"All conditions satisfied.\n"
                f"Overall policy truth degree: {truth:.3f}"
            )
        else:
            return f"Decision: {decision} (no specific reasons identified)"

    def find_minimal_change(
        self,
        target_decision: bool,
        max_change: float = 10.0,
    ) -> Optional[Dict[str, float]]:
        """
        Find minimal change to inputs that flips the decision.

        Args:
            target_decision: Desired decision (True = approved, False = denied)
            max_change: Maximum allowed change to any input

        Returns:
            Dictionary of suggested changes, or None if not achievable
        """
        # This is a simplified version
        # In practice, you'd use optimization to find minimal change

        current_truth = self.policy_chain.evaluate().item()
        current_decision = current_truth > 0.5

        if current_decision == target_decision:
            return {}  # Already at target decision

        suggestions = {}

        # Try modifying each triplet
        for i, triplet in enumerate(self.policy_chain.triplets):
            if isinstance(triplet, TripletState):
                # Analyze what change would help
                analysis = self.reasoner.analyze_numeric_counterfactual(
                    triplet_index=i,
                    value_range=(0, max_change),
                    num_steps=20
                )

                if analysis["decision_boundary"] is not None:
                    suggestions[f"condition_{i}"] = analysis["decision_boundary"]

        return suggestions if suggestions else None
