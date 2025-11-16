"""
Clifford Algebra Cl(3,0) Implementation

Implements the Euclidean geometric algebra Cl(3,0) for 3D space with signature (+,+,+).
Provides differentiable geometric operations for neurosymbolic reasoning.

Basis elements:
    - Scalars: 1
    - Vectors: e1, e2, e3
    - Bivectors: e12, e13, e23
    - Trivector: e123

A multivector in Cl(3,0) is represented as an 8-dimensional tensor:
    M = [scalar, e1, e2, e3, e12, e13, e23, e123]
"""

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThresholdMode(Enum):
    """Threshold evaluation modes for numeric predicates"""
    HARD = "hard"      # Binary: 1 if value >= threshold else 0
    SOFT = "soft"      # Sigmoid: smooth transition
    MARGIN = "margin"  # Margin-based: threshold - margin


class Multivector(nn.Module):
    """
    Multivector in Cl(3,0) represented as 8D tensor.

    Components: [scalar, e1, e2, e3, e12, e13, e23, e123]

    Args:
        components: 8D tensor or array of multivector components
        requires_grad: Whether to track gradients (default: True)
    """

    def __init__(
        self,
        components: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
    ):
        super().__init__()

        if components is None:
            components = torch.zeros(8)

        if not isinstance(components, torch.Tensor):
            components = torch.tensor(components, dtype=torch.float32)

        self.components = nn.Parameter(components, requires_grad=requires_grad)
        self.epsilon = 1e-8

    def scalar(self) -> torch.Tensor:
        """Extract scalar part"""
        return self.components[0]

    def vector(self) -> torch.Tensor:
        """Extract vector part [e1, e2, e3]"""
        return self.components[1:4]

    def bivector(self) -> torch.Tensor:
        """Extract bivector part [e12, e13, e23]"""
        return self.components[4:7]

    def trivector(self) -> torch.Tensor:
        """Extract trivector part (pseudoscalar)"""
        return self.components[7]

    def norm(self) -> torch.Tensor:
        """Euclidean norm of multivector"""
        return torch.sqrt(torch.sum(self.components ** 2) + self.epsilon)

    def normalize(self) -> "Multivector":
        """Return normalized multivector"""
        return Multivector(self.components / self.norm())

    def reverse(self) -> "Multivector":
        """
        Clifford reverse (reversion)
        Reverses order of basis vectors in products

        For Cl(3,0):
            scalar → scalar
            vector → vector
            bivector → -bivector
            trivector → -trivector
        """
        rev_components = self.components.clone()
        rev_components[4:8] = -rev_components[4:8]  # Negate bivector and trivector
        return Multivector(rev_components)

    def __repr__(self) -> str:
        c = self.components.detach()
        return (
            f"Multivector({c[0]:.3f} + {c[1]:.3f}e1 + {c[2]:.3f}e2 + {c[3]:.3f}e3 "
            f"+ {c[4]:.3f}e12 + {c[5]:.3f}e13 + {c[6]:.3f}e23 + {c[7]:.3f}e123)"
        )


class Vector(Multivector):
    """
    Vector in R³ embedded in Cl(3,0)

    Args:
        x, y, z: Vector components along e1, e2, e3
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        components = torch.tensor([0.0, x, y, z, 0.0, 0.0, 0.0, 0.0])
        super().__init__(components)

    @classmethod
    def from_tensor(cls, vec: torch.Tensor) -> "Vector":
        """Create vector from 3D tensor [x, y, z]"""
        if vec.shape[-1] != 3:
            raise ValueError(f"Expected 3D vector, got shape {vec.shape}")
        components = torch.zeros(8)
        components[1:4] = vec
        return Vector.__new__(Vector)


class Bivector(Multivector):
    """
    Bivector in Cl(3,0) representing oriented plane

    Args:
        e12, e13, e23: Bivector components
    """

    def __init__(self, e12: float = 0.0, e13: float = 0.0, e23: float = 0.0):
        components = torch.tensor([0.0, 0.0, 0.0, 0.0, e12, e13, e23, 0.0])
        super().__init__(components)

    @classmethod
    def from_vectors(cls, u: torch.Tensor, v: torch.Tensor) -> "Bivector":
        """
        Construct bivector from wedge product u ∧ v

        u ∧ v = (u1e1 + u2e2 + u3e3) ∧ (v1e1 + v2e2 + v3e3)
              = (u1v2 - u2v1)e12 + (u1v3 - u3v1)e13 + (u2v3 - u3v2)e23
        """
        if u.shape[-1] != 3 or v.shape[-1] != 3:
            raise ValueError("Expected 3D vectors")

        e12 = u[0] * v[1] - u[1] * v[0]
        e13 = u[0] * v[2] - u[2] * v[0]
        e23 = u[1] * v[2] - u[2] * v[1]

        return Bivector(e12.item(), e13.item(), e23.item())


class Rotor(Multivector):
    """
    Rotor in Cl(3,0) representing rotation

    A rotor R = cos(θ/2) + sin(θ/2)B where B is a unit bivector
    defines a rotation by angle θ in the plane represented by B.

    Rotors act on vectors via the sandwich product: v' = RvR†

    Args:
        bivector: Unit bivector defining rotation plane
        angle: Rotation angle in radians
    """

    def __init__(self, bivector: Bivector, angle: float):
        # R = cos(θ/2) + sin(θ/2)B
        # First normalize the bivector
        B_components = bivector.bivector()
        B_norm = torch.sqrt(torch.sum(B_components ** 2) + 1e-8)
        B_unit = B_components / B_norm

        half_angle = angle / 2.0
        cos_half = torch.cos(torch.tensor(half_angle))
        sin_half = torch.sin(torch.tensor(half_angle))

        components = torch.zeros(8)
        components[0] = cos_half
        components[4:7] = sin_half * B_unit

        super().__init__(components, requires_grad=True)

    @classmethod
    def identity(cls) -> "Rotor":
        """Identity rotor (no rotation)"""
        return Rotor(Bivector(0, 0, 1), 0.0)

    @classmethod
    def from_angle_bivector(cls, angle: float, bivector: Bivector) -> "Rotor":
        """Construct rotor from angle and bivector"""
        return cls(bivector, angle)

    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Apply rotor to vector via sandwich product: v' = RvR†

        Args:
            vector: 3D vector tensor [e1, e2, e3]

        Returns:
            Rotated 3D vector
        """
        # Create multivector from vector
        v_multi = torch.zeros(8)
        v_multi[1:4] = vector

        # Compute RvR† using geometric product
        Rv = geometric_product(self.components, v_multi)
        R_reverse = self.reverse().components
        result = geometric_product(Rv, R_reverse)

        # Extract vector part
        return result[1:4]


def geometric_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Geometric product of two multivectors in Cl(3,0)

    The geometric product combines both inner and outer products:
        AB = A·B + A∧B

    Args:
        A, B: 8D multivector tensors

    Returns:
        8D multivector result
    """
    # Extract components
    a0, a1, a2, a3, a12, a13, a23, a123 = A.unbind(-1) if A.dim() > 0 else A
    b0, b1, b2, b3, b12, b13, b23, b123 = B.unbind(-1) if B.dim() > 0 else B

    # Compute geometric product (multiplication table for Cl(3,0))
    result = torch.zeros(8)

    # Scalar part
    result[0] = (a0*b0 + a1*b1 + a2*b2 + a3*b3
                 - a12*b12 - a13*b13 - a23*b23 - a123*b123)

    # Vector parts
    result[1] = (a0*b1 + a1*b0 - a12*b2 - a13*b3
                 + a2*b12 + a3*b13 - a23*b123 - a123*b23)
    result[2] = (a0*b2 + a2*b0 + a12*b1 - a23*b3
                 - a1*b12 + a3*b23 + a13*b123 + a123*b13)
    result[3] = (a0*b3 + a3*b0 + a13*b1 + a23*b2
                 - a1*b13 - a2*b23 - a12*b123 - a123*b12)

    # Bivector parts
    result[4] = (a0*b12 + a12*b0 + a1*b2 - a2*b1
                 + a123*b3 + a3*b123 - a13*b23 + a23*b13)
    result[5] = (a0*b13 + a13*b0 + a1*b3 - a3*b1
                 - a123*b2 - a2*b123 + a12*b23 - a23*b12)
    result[6] = (a0*b23 + a23*b0 + a2*b3 - a3*b2
                 + a123*b1 + a1*b123 - a12*b13 + a13*b12)

    # Trivector part
    result[7] = (a0*b123 + a123*b0 + a1*b23 - a2*b13 + a3*b12
                 + a12*b3 - a13*b2 + a23*b1)

    return result


def sandwich_product(rotor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """
    Sandwich product: v' = RvR†

    Args:
        rotor: 8D rotor multivector
        vector: 3D vector

    Returns:
        Rotated 3D vector
    """
    # Convert vector to multivector
    v_multi = torch.zeros(8)
    v_multi[1:4] = vector

    # Compute reverse of rotor
    rotor_rev = rotor.clone()
    rotor_rev[4:8] = -rotor_rev[4:8]

    # RvR†
    Rv = geometric_product(rotor, v_multi)
    result = geometric_product(Rv, rotor_rev)

    # Extract vector part
    return result[1:4]


def inner_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Inner product (contraction) of two vectors

    For vectors u, v in R³:
        u·v = u1v1 + u2v2 + u3v3 (scalar)

    Args:
        A, B: 3D vector tensors

    Returns:
        Scalar inner product
    """
    return torch.sum(A * B, dim=-1)


def wedge_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wedge (outer) product of two vectors producing bivector

    For vectors u, v in R³:
        u∧v = (u1v2 - u2v1)e12 + (u1v3 - u3v1)e13 + (u2v3 - u3v2)e23

    Args:
        A, B: 3D vector tensors

    Returns:
        3D bivector tensor [e12, e13, e23]
    """
    e12 = A[0] * B[1] - A[1] * B[0]
    e13 = A[0] * B[2] - A[2] * B[0]
    e23 = A[1] * B[2] - A[2] * B[1]

    return torch.stack([e12, e13, e23])
