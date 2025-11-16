"""Geometric Algebra Module for NeuraLog.

This module implements Clifford/Geometric Algebra (Cl(3,0)) components for
symbolic reasoning with neural-symbolic integration.

Core Components:
- TripletLogicLayer: Core GA layer for subject-predicate-object triplets
- Multivector operations: Geometric products and grades
- Truth degree computation: From canonical vector states
- Logical connectives: AND, OR, implication, equivalence
- Policy framework: Define and verify policies using GA triplets

High-Performance Components:
- BatchedPolicyVerifier: GPU-optimized batched verification
- CounterfactualReasoner: Gradient-based policy refinement
- VerificationMetrics: Accuracy, calibration, and performance tracking
- NeuraLogGASystem: Unified end-to-end reasoning engine
- PolicyExtractorPipeline: LLM-integrated extraction with policy verification
"""

from neuralog.symbolic.geometric_algebra.batched_verifier import (
    BatchedPolicyVerifier,
    BatchVerificationResult,
    VerificationResult,
)
from neuralog.symbolic.geometric_algebra.counterfactual_reasoner import (
    CounterfactualExplanation,
    CounterfactualReasoner,
)
from neuralog.symbolic.geometric_algebra.multivector import Multivector
from neuralog.symbolic.geometric_algebra.neuralog_system import (
    NeuraLogGASystem,
    PolicyVerificationResult,
    SystemConfig,
)
from neuralog.symbolic.geometric_algebra.policy import (
    Policy,
    PolicyCondition,
    PolicyRule,
)
from neuralog.symbolic.geometric_algebra.policy_extractor_pipeline import (
    ExtractedFact,
    PolicyExtractorPipeline,
    RefinementFeedback,
)
from neuralog.symbolic.geometric_algebra.triplet_logic import (
    ThresholdMode,
    TripletLogicLayer,
)
from neuralog.symbolic.geometric_algebra.verification_metrics import (
    CalibrationMetrics,
    ConfusionMatrices,
    LatencyMetrics,
    MetricsCollector,
    ThroughputMetrics,
    VerificationMetrics,
)

__all__ = [
    # Core
    "TripletLogicLayer",
    "ThresholdMode",
    "Multivector",
    "PolicyCondition",
    "PolicyRule",
    "Policy",
    # High-performance
    "BatchedPolicyVerifier",
    "BatchVerificationResult",
    "VerificationResult",
    "CounterfactualReasoner",
    "CounterfactualExplanation",
    "VerificationMetrics",
    "MetricsCollector",
    "CalibrationMetrics",
    "ConfusionMatrices",
    "LatencyMetrics",
    "ThroughputMetrics",
    # Unified system
    "NeuraLogGASystem",
    "PolicyVerificationResult",
    "SystemConfig",
    # LLM Integration
    "PolicyExtractorPipeline",
    "ExtractedFact",
    "RefinementFeedback",
]
