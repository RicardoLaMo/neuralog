"""
Triplet Logic Layer

Implements logical reasoning over knowledge graph triplets using geometric algebra.
Each triplet (subject, predicate, object) is encoded as a geometric structure in Cl(3,0),
either as a rotor or as a canonical state vector.

Key concepts:
- TripletState: Canonical encoding using 3 axes (subject, predicate, truth)
- TripletRotor: General encoding as rotation in subject-object plane
- Truth degrees: Geometric alignment measures
- Logical operations: AND, OR, NOT via geometric composition
"""

from enum import Enum
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clifford import (
    Multivector,
    Vector,
    Bivector,
    Rotor,
    geometric_product,
    sandwich_product,
    wedge_product,
    inner_product,
    ThresholdMode,
)


class TripletState(nn.Module):
    """
    Canonical triplet representation using 3-axis encoding.

    Axes:
        e1: Subject axis (degree of subject presence)
        e2: Predicate axis (strength of relationship)
        e3: Truth/value axis (truth value or numeric value)

    State vector:
        x_τ = x_s·e1 + x_p·e2 + x_t·e3

    Args:
        x_s: Subject component [0, 1]
        x_p: Predicate component [0, 1]
        x_t: Truth/value component (normalized or raw value)
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        x_s: float = 1.0,
        x_p: float = 1.0,
        x_t: float = 1.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        # Create state vector [e1, e2, e3]
        self.state = nn.Parameter(
            torch.tensor([x_s, x_p, x_t], dtype=torch.float32),
            requires_grad=True
        )
        self.epsilon = epsilon

    def truth_degree(self) -> torch.Tensor:
        """
        Compute truth degree from canonical state.

        truth(τ) = |x_t| / sqrt(x_s² + x_p² + x_t²)

        Returns:
            Truth degree in [0, 1]
        """
        x_s, x_p, x_t = self.state.unbind(-1) if self.state.dim() > 0 else self.state

        norm = torch.sqrt(x_s**2 + x_p**2 + x_t**2 + self.epsilon)
        return torch.abs(x_t) / norm

    def to_multivector(self) -> Multivector:
        """Convert to multivector representation in Cl(3,0)"""
        components = torch.zeros(8)
        components[1:4] = self.state  # Vector part [e1, e2, e3]
        return Multivector(components)

    def __repr__(self) -> str:
        s = self.state.detach()
        t = self.truth_degree().item()
        return f"TripletState(s={s[0]:.3f}, p={s[1]:.3f}, t={s[2]:.3f}, truth={t:.3f})"


class TripletRotor(nn.Module):
    """
    General triplet representation as rotor in Cl(3,0).

    Encodes triplet τ = (subject, predicate, object) as:
        R_τ = cos(θ/2) + sin(θ/2)B

    Where:
        B: Bivector in plane from subject to object
        θ: Angle encoding predicate strength

    Args:
        subject: 3D vector representing subject entity
        predicate_strength: Scalar in [0, 1] encoding relationship strength
        object_vec: 3D vector representing object entity
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        subject: torch.Tensor,
        predicate_strength: float,
        object_vec: torch.Tensor,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        self.epsilon = epsilon

        # Normalize vectors
        self.subject = nn.Parameter(
            subject / (torch.norm(subject) + epsilon),
            requires_grad=True
        )
        self.object = nn.Parameter(
            object_vec / (torch.norm(object_vec) + epsilon),
            requires_grad=True
        )

        # Predicate strength determines rotation angle
        self.predicate_strength = nn.Parameter(
            torch.tensor(predicate_strength, dtype=torch.float32),
            requires_grad=True
        )

        # Build rotor
        self._construct_rotor()

    def _construct_rotor(self):
        """Construct rotor from subject, object, and predicate strength"""
        # Compute bivector B = (subject ∧ object) / ||subject ∧ object||
        bivec = wedge_product(self.subject, self.object)
        bivec_norm = torch.sqrt(torch.sum(bivec**2) + self.epsilon)

        # Handle degenerate case (colinear vectors)
        if bivec_norm < 1e-6:
            # Use canonical e13 plane if vectors are colinear
            bivec = torch.tensor([0.0, 1.0, 0.0])  # e13 bivector
            bivec_norm = 1.0

        B_unit = bivec / bivec_norm

        # Angle from predicate strength: θ = π·σ(predicate_strength)
        theta = torch.pi * torch.sigmoid(self.predicate_strength)

        # Construct rotor components: cos(θ/2) + sin(θ/2)B
        half_theta = theta / 2.0
        cos_half = torch.cos(half_theta)
        sin_half = torch.sin(half_theta)

        # Rotor multivector [scalar, 0, 0, 0, e12, e13, e23, 0]
        rotor_components = torch.zeros(8)
        rotor_components[0] = cos_half
        rotor_components[4:7] = sin_half * B_unit

        self.rotor = Rotor.__new__(Rotor)
        self.rotor.components = nn.Parameter(rotor_components, requires_grad=True)
        self.rotor.epsilon = self.epsilon

    def truth_degree(self) -> torch.Tensor:
        """
        Compute truth degree via rotor-mediated alignment.

        truth(τ) = (R_τ·s·R_τ†)·o / (||R_τ·s·R_τ†|| ||o||)

        Returns:
            Truth degree in [-1, 1] (map to [0, 1] with (1+t)/2 if needed)
        """
        # Apply rotor to subject: s' = R·s·R†
        s_rotated = sandwich_product(self.rotor.components, self.subject)

        # Compute alignment with object
        dot_product = inner_product(s_rotated, self.object)
        norm_s_rotated = torch.norm(s_rotated) + self.epsilon
        norm_object = torch.norm(self.object) + self.epsilon

        return dot_product / (norm_s_rotated * norm_object)

    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply rotor to vector via sandwich product"""
        return self.rotor.apply(vector)

    def __repr__(self) -> str:
        t = self.truth_degree().item()
        p = torch.sigmoid(self.predicate_strength).item()
        return f"TripletRotor(predicate_strength={p:.3f}, truth={t:.3f})"


class TripletEncoder(nn.Module):
    """
    Encoder for converting triplets to geometric representations.

    Supports two modes:
        - canonical: Use fixed 3-axis encoding (simpler, interpretable)
        - rotor: Use general rotor encoding (more expressive)

    Args:
        mode: Encoding mode ("canonical" or "rotor")
        threshold_mode: How to evaluate numeric thresholds
        epsilon: Numerical stability constant
    """

    def __init__(
        self,
        mode: str = "canonical",
        threshold_mode: ThresholdMode = ThresholdMode.HARD,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        if mode not in ["canonical", "rotor"]:
            raise ValueError(f"Unknown mode: {mode}. Use 'canonical' or 'rotor'")

        self.mode = mode
        self.threshold_mode = threshold_mode
        self.epsilon = epsilon

    def encode_numeric_triplet(
        self,
        value: float,
        threshold: float,
        subject_strength: float = 1.0,
        predicate_strength: float = 1.0,
    ) -> TripletState:
        """
        Encode numeric comparison triplet (e.g., age >= 65).

        Args:
            value: Numeric value to compare
            threshold: Threshold for comparison
            subject_strength: Subject presence [0, 1]
            predicate_strength: Predicate strength [0, 1]

        Returns:
            TripletState with truth value based on threshold mode
        """
        # Compute truth component based on threshold mode
        if self.threshold_mode == ThresholdMode.HARD:
            x_t = 1.0 if value >= threshold else 0.0
        elif self.threshold_mode == ThresholdMode.SOFT:
            # Sigmoid with steep slope (β=20)
            x_t = torch.sigmoid(torch.tensor(20.0 * (value - threshold))).item()
        elif self.threshold_mode == ThresholdMode.MARGIN:
            # Margin mode: effective threshold is threshold - margin
            margin = 0.5
            x_t = torch.sigmoid(torch.tensor(20.0 * (value - threshold + margin))).item()
        else:
            raise ValueError(f"Unknown threshold mode: {self.threshold_mode}")

        return TripletState(
            x_s=subject_strength,
            x_p=predicate_strength,
            x_t=x_t,
            epsilon=self.epsilon
        )

    def encode_binary_triplet(
        self,
        truth_value: bool,
        subject_strength: float = 1.0,
        predicate_strength: float = 1.0,
    ) -> TripletState:
        """
        Encode binary truth triplet (e.g., isSenior = True).

        Args:
            truth_value: Boolean value
            subject_strength: Subject presence [0, 1]
            predicate_strength: Predicate strength [0, 1]

        Returns:
            TripletState
        """
        x_t = 1.0 if truth_value else 0.0
        return TripletState(
            x_s=subject_strength,
            x_p=predicate_strength,
            x_t=x_t,
            epsilon=self.epsilon
        )

    def encode_rotor_triplet(
        self,
        subject_vec: torch.Tensor,
        predicate_strength: float,
        object_vec: torch.Tensor,
    ) -> TripletRotor:
        """
        Encode triplet as rotor.

        Args:
            subject_vec: 3D subject vector
            predicate_strength: Predicate strength [0, 1]
            object_vec: 3D object vector

        Returns:
            TripletRotor
        """
        return TripletRotor(
            subject=subject_vec,
            predicate_strength=predicate_strength,
            object_vec=object_vec,
            epsilon=self.epsilon
        )


def truth_degree(triplet: Union[TripletState, TripletRotor]) -> torch.Tensor:
    """
    Compute truth degree of a triplet.

    Args:
        triplet: TripletState or TripletRotor

    Returns:
        Truth degree tensor in [0, 1] (or [-1, 1] for rotors)
    """
    return triplet.truth_degree()


def logical_and(
    triplets: List[Union[TripletState, TripletRotor]],
    method: str = "min",
) -> torch.Tensor:
    """
    Logical AND over multiple triplets.

    Methods:
        - "min": Minimum t-norm (default)
        - "product": Product t-norm
        - "lukasiewicz": Łukasiewicz t-norm

    Args:
        triplets: List of triplet states/rotors
        method: T-norm method

    Returns:
        Combined truth degree
    """
    truths = [truth_degree(t) for t in triplets]

    if method == "min":
        return torch.min(torch.stack(truths))
    elif method == "product":
        result = truths[0]
        for t in truths[1:]:
            result = result * t
        return result
    elif method == "lukasiewicz":
        result = truths[0]
        for t in truths[1:]:
            result = torch.max(torch.tensor(0.0), result + t - 1.0)
        return result
    else:
        raise ValueError(f"Unknown method: {method}")


def logical_or(
    triplets: List[Union[TripletState, TripletRotor]],
    method: str = "probabilistic",
) -> torch.Tensor:
    """
    Logical OR over multiple triplets.

    Methods:
        - "probabilistic": Probabilistic sum (default)
        - "max": Maximum t-conorm
        - "lukasiewicz": Łukasiewicz t-conorm

    Args:
        triplets: List of triplet states/rotors
        method: T-conorm method

    Returns:
        Combined truth degree
    """
    truths = [truth_degree(t) for t in triplets]

    if method == "probabilistic":
        # t1 + t2 - t1·t2
        result = truths[0]
        for t in truths[1:]:
            result = result + t - result * t
        return result
    elif method == "max":
        return torch.max(torch.stack(truths))
    elif method == "lukasiewicz":
        result = truths[0]
        for t in truths[1:]:
            result = torch.min(torch.tensor(1.0), result + t)
        return result
    else:
        raise ValueError(f"Unknown method: {method}")


def logical_not(triplet: Union[TripletState, TripletRotor]) -> torch.Tensor:
    """
    Logical NOT of a triplet.

    For canonical: NOT(t) = 1 - t
    For rotor: Invert rotor (R† instead of R)

    Args:
        triplet: TripletState or TripletRotor

    Returns:
        Negated truth degree
    """
    t = truth_degree(triplet)

    if isinstance(triplet, TripletState):
        # Simple complement for canonical encoding
        return 1.0 - t
    else:
        # For rotors, we'd invert the rotor, but for truth degree just complement
        # (Full rotor inversion would require reconstructing the rotor)
        return 1.0 - ((t + 1.0) / 2.0)  # Map [-1,1] to [0,1], complement, map back


def implication_loss(
    antecedent: Union[TripletState, TripletRotor],
    consequent: Union[TripletState, TripletRotor],
) -> torch.Tensor:
    """
    Differentiable loss for implication: antecedent ⇒ consequent

    L_⇒ = max(0, truth(antecedent) - truth(consequent))²

    Penalizes cases where antecedent is true but consequent is false.

    Args:
        antecedent: Antecedent triplet
        consequent: Consequent triplet

    Returns:
        Implication loss (0 if satisfied, >0 if violated)
    """
    t_ant = truth_degree(antecedent)
    t_con = truth_degree(consequent)

    # Normalize to [0, 1] if needed (for rotors)
    if isinstance(antecedent, TripletRotor):
        t_ant = (t_ant + 1.0) / 2.0
    if isinstance(consequent, TripletRotor):
        t_con = (t_con + 1.0) / 2.0

    violation = torch.max(torch.tensor(0.0), t_ant - t_con)
    return violation ** 2


def equivalence_loss(
    triplet1: Union[TripletState, TripletRotor],
    triplet2: Union[TripletState, TripletRotor],
) -> torch.Tensor:
    """
    Differentiable loss for equivalence: triplet1 ⇔ triplet2

    L_⇔ = (truth(triplet1) - truth(triplet2))²

    Enforces bidirectional equality.

    Args:
        triplet1: First triplet
        triplet2: Second triplet

    Returns:
        Equivalence loss (0 if equal, >0 if different)
    """
    t1 = truth_degree(triplet1)
    t2 = truth_degree(triplet2)

    # Normalize to [0, 1] if needed
    if isinstance(triplet1, TripletRotor):
        t1 = (t1 + 1.0) / 2.0
    if isinstance(triplet2, TripletRotor):
        t2 = (t2 + 1.0) / 2.0

    return (t1 - t2) ** 2


def rotor_consistency_loss(
    triplets: List[TripletRotor],
    target_rotor: Rotor,
) -> torch.Tensor:
    """
    Loss for rotor chain consistency.

    L_rotor = ||R_n···R_2·R_1 - R_target||²

    Ensures composition of triplet rotors matches target.

    Args:
        triplets: List of triplet rotors to compose
        target_rotor: Target rotor

    Returns:
        Rotor consistency loss
    """
    if not triplets:
        return torch.tensor(0.0)

    # Compose rotors via geometric product
    result = triplets[0].rotor.components
    for triplet in triplets[1:]:
        result = geometric_product(result, triplet.rotor.components)

    # Compute distance to target
    diff = result - target_rotor.components
    return torch.sum(diff ** 2)
