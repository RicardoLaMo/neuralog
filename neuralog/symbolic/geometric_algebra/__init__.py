"""Geometric Algebra Module for NeuraLog.

This module implements Clifford/Geometric Algebra (Cl(3,0)) components for
symbolic reasoning with neural-symbolic integration.

Key Components:
- TripletLogicLayer: Core GA layer for subject-predicate-object triplets
- Multivector operations: Geometric products and grades
- Truth degree computation: From canonical vector states
- Logical connectives: AND, OR, implication, equivalence
- Policy framework: Define and verify policies using GA triplets
"""

from neuralog.symbolic.geometric_algebra.multivector import Multivector
from neuralog.symbolic.geometric_algebra.policy import (
    Policy,
    PolicyCondition,
    PolicyRule,
)
from neuralog.symbolic.geometric_algebra.triplet_logic import (
    ThresholdMode,
    TripletLogicLayer,
)

__all__ = [
    "TripletLogicLayer",
    "ThresholdMode",
    "Multivector",
    "PolicyCondition",
    "PolicyRule",
    "Policy",
]
