"""
Geometric Algebra Module for NeuraLog

Implements geometric algebra (GA) Cl(3,0) for differentiable logical reasoning
over knowledge graph triplets. Based on the triplet decomposition principle where
complex logical expressions are represented as chains of subject-predicate-object
(SPO) triplets, each encoded as a rotor in geometric algebra.

Key concepts:
- Multivectors in Cl(3,0): scalars, vectors, bivectors, trivectors
- Rotors: geometric transformations representing triplet relationships
- Truth degrees: geometric alignment via rotor application
- Differentiable logic: gradient-based policy enforcement

References:
    Geometric view of neural logic and neurosymbolic verification
    using triplet decomposition in Clifford algebra Cl(3,0).
"""

from .clifford import (
    Multivector,
    Vector,
    Bivector,
    Rotor,
    geometric_product,
    sandwich_product,
    inner_product,
    wedge_product,
    ThresholdMode,
)

from .triplet_logic import (
    TripletState,
    TripletRotor,
    TripletEncoder,
    truth_degree,
    logical_and,
    logical_or,
    logical_not,
    implication_loss,
    equivalence_loss,
    rotor_consistency_loss,
)

from .policy_encoder import (
    PolicyEncoder,
    PolicyChain,
    PolicyValidator,
    VerificationMode,
    verify_llm_answer,
    verify_policy_chain,
    build_policy_from_triplets,
)

from .counterfactual import (
    CounterfactualReasoner,
    CounterfactualExplainer,
    rotor_trajectory,
    counterfactual_robustness_loss,
)

__all__ = [
    # Core GA
    "Multivector",
    "Vector",
    "Bivector",
    "Rotor",
    "geometric_product",
    "sandwich_product",
    "inner_product",
    "wedge_product",
    "ThresholdMode",
    # Triplet logic
    "TripletState",
    "TripletRotor",
    "TripletEncoder",
    "truth_degree",
    "logical_and",
    "logical_or",
    "logical_not",
    "implication_loss",
    "equivalence_loss",
    "rotor_consistency_loss",
    # Policy encoding
    "PolicyEncoder",
    "PolicyChain",
    "PolicyValidator",
    "VerificationMode",
    "verify_llm_answer",
    "verify_policy_chain",
    "build_policy_from_triplets",
    # Counterfactual reasoning
    "CounterfactualReasoner",
    "CounterfactualExplainer",
    "rotor_trajectory",
    "counterfactual_robustness_loss",
]
