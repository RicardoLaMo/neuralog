"""Triplet Logic Layer for Geometric Algebra.

Implements the canonical triplet state representation and truth degree computation
for subject-predicate-object triplets in Cl(3,0) (Euclidean Geometric Algebra).

Based on Section 7.1 of the GACoreCl30 architecture:
- Canonical state: x_τ = [0, x_s, x_p, x_t, 0, 0, 0, 0]
- Truth degree: |x_t| / ||[x_s, x_p, x_t]||
- Threshold modes: HARD, SOFT, MARGIN for numeric predicates
"""

from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn


class ThresholdMode(str, Enum):
    """Threshold modes for numeric predicate evaluation."""

    HARD = "hard"  # Binary step function
    SOFT = "soft"  # Sigmoid activation
    MARGIN = "margin"  # Sigmoid with margin


class TripletLogicLayer(nn.Module):
    """Triplet Logic Layer for symbolic reasoning in Geometric Algebra.

    Maps subject-predicate-object triplets to Cl(3,0) multivectors with
    canonical vector state representation and truth degree computation.

    Attributes:
        eps: Small value for numerical stability
        beta: Steepness of sigmoid for SOFT and MARGIN threshold modes
        margin: Margin parameter for MARGIN threshold mode
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        beta: float = 20.0,
        margin: float = 0.5,
    ):
        """Initialize TripletLogicLayer.

        Args:
            epsilon: Numerical stability constant. Defaults to 1e-8.
            beta: Sigmoid steepness parameter. Defaults to 20.0.
            margin: Margin for MARGIN mode. Defaults to 0.5.
        """
        super().__init__()
        self.eps = epsilon
        self.beta = beta
        self.margin = margin

    @staticmethod
    def create_vector_state(
        x_s: torch.Tensor,
        x_p: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Create canonical vector state in Cl(3,0).

        Creates a multivector with only the vector part filled:
        x_τ = [0, x_s, x_p, x_t, 0, 0, 0, 0]

        Where:
        - 0 (scalar part)
        - x_s (e1 component: subject)
        - x_p (e2 component: predicate)
        - x_t (e3 component: truth/object)
        - 0, 0, 0 (bivector parts)
        - 0 (pseudoscalar)

        Args:
            x_s: Subject component [...,]
            x_p: Predicate component [...,]
            x_t: Truth/object component [...,]

        Returns:
            Multivector in canonical form [..., 8] with structure:
            [scalar, e1, e2, e3, e12, e13, e23, e123]
        """
        zeros = torch.zeros_like(x_s)
        return torch.stack(
            [zeros, x_s, x_p, x_t, zeros, zeros, zeros, zeros],
            dim=-1,
        )

    def numeric_triplet_state(
        self,
        value: torch.Tensor,
        threshold: torch.Tensor,
        mode: ThresholdMode = ThresholdMode.SOFT,
    ) -> torch.Tensor:
        """Convert numeric predicate to triplet state.

        Maps a numeric value and threshold to canonical vector state
        using the specified threshold mode.

        Args:
            value: Numeric value (e.g., age, budget) [...,]
            threshold: Threshold value [...,]
            mode: Threshold evaluation mode. Defaults to SOFT.

        Returns:
            Canonical vector state [..., 8]

        Raises:
            ValueError: If mode is not recognized.
        """
        # Normalize value relative to threshold
        x1 = value / (threshold.clamp(min=self.eps) * 2.0)
        x2 = torch.ones_like(value)

        # Apply threshold mode
        if mode == ThresholdMode.HARD:
            x3 = (value >= threshold).float()
        elif mode == ThresholdMode.SOFT:
            x3 = torch.sigmoid(self.beta * (value - threshold))
        elif mode == ThresholdMode.MARGIN:
            x3 = torch.sigmoid(self.beta * (value - threshold + self.margin))
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")

        return self.create_vector_state(x1, x2, x3)

    def truth_degree(self, x_tau: torch.Tensor) -> torch.Tensor:
        """Compute truth degree from canonical vector state.

        Implements Eq. (23): truth_degree = |x_t| / ||[x_s, x_p, x_t]||

        Where the norm is computed over the vector part (e1, e2, e3 components).

        Args:
            x_tau: Canonical vector state [..., 8]

        Returns:
            Truth degree in [0, 1] [...,]
        """
        # Extract vector part (indices 1, 2, 3 = e1, e2, e3)
        vec_part = x_tau[..., 1:4]  # [..., 3]
        x_t = x_tau[..., 3]  # e3 component (truth)

        # Compute norm of [x_s, x_p, x_t]
        norm = torch.linalg.norm(vec_part, dim=-1)

        # Return |x_t| / norm
        return torch.abs(x_t) / (norm + self.eps)

    # Logical connectives on truth degrees

    @staticmethod
    def t_and(*t_list: torch.Tensor) -> torch.Tensor:
        """Fuzzy AND operation on truth degrees.

        Implements minimum t-norm: t_and(t1, t2, ...) = min(t1, t2, ...)

        Args:
            *t_list: Variable number of truth degree tensors

        Returns:
            Minimum truth degree across all inputs
        """
        t = torch.stack(t_list, dim=-1)
        return torch.min(t, dim=-1).values

    @staticmethod
    def t_or(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Fuzzy OR operation on truth degrees.

        Implements probabilistic t-conorm: t_or(t1, t2) = t1 + t2 - t1*t2

        Args:
            t1: First truth degree
            t2: Second truth degree

        Returns:
            Fuzzy OR of the two truth degrees
        """
        return t1 + t2 - t1 * t2

    @staticmethod
    def loss_implies(t_p: torch.Tensor, t_q: torch.Tensor) -> torch.Tensor:
        """Loss for implication p ⇒ q.

        Implements penalty for violated implications:
        L(p ⇒ q) = max(0, t_p - t_q)^2

        Args:
            t_p: Antecedent truth degree
            t_q: Consequent truth degree

        Returns:
            Implication loss
        """
        return torch.clamp(t_p - t_q, min=0.0) ** 2

    @staticmethod
    def loss_equiv(t_p: torch.Tensor, t_q: torch.Tensor) -> torch.Tensor:
        """Loss for equivalence p ≡ q.

        Implements squared difference penalty:
        L(p ≡ q) = (t_p - t_q)^2

        Args:
            t_p: First truth degree
            t_q: Second truth degree

        Returns:
            Equivalence loss
        """
        return (t_p - t_q) ** 2

    @staticmethod
    def loss_bidirectional(
        t_llm: torch.Tensor,
        t_policy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute bidirectional loss for LLM-Policy comparison.

        Separates false positives and false negatives:
        - FP: LLM claims true but policy says false (t_llm > t_policy)
        - FN: Policy claims true but LLM misses (t_policy > t_llm)

        Args:
            t_llm: LLM extracted truth degree
            t_policy: Policy-derived truth degree

        Returns:
            Tuple of (false_positive_loss, false_negative_loss)
        """
        fp = torch.clamp(t_llm - t_policy, min=0.0) ** 2
        fn = torch.clamp(t_policy - t_llm, min=0.0) ** 2
        return fp, fn

    def forward(
        self,
        x_tau: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: compute truth degree from canonical state.

        Args:
            x_tau: Canonical vector state [..., 8]

        Returns:
            Truth degree [...,]
        """
        return self.truth_degree(x_tau)
