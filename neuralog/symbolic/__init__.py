"""Symbolic processing components."""

from neuralog.symbolic.graph_embedder import GraphEmbedder
from neuralog.symbolic.ontology_manager import OntologyManager
from neuralog.symbolic.reasoner import SymbolicReasoner

# Geometric Algebra components
from neuralog.symbolic.geometric_algebra import (
    Multivector,
    Policy,
    PolicyCondition,
    PolicyRule,
    ThresholdMode,
    TripletLogicLayer,
)
from neuralog.symbolic.geometric_algebra.ga_reasoner import (
    GATripleEvaluation,
    GeometricAlgebraReasoner,
)

__all__ = [
    "OntologyManager",
    "GraphEmbedder",
    "SymbolicReasoner",
    # Geometric Algebra
    "TripletLogicLayer",
    "ThresholdMode",
    "Multivector",
    "PolicyCondition",
    "PolicyRule",
    "Policy",
    "GeometricAlgebraReasoner",
    "GATripleEvaluation",
]
