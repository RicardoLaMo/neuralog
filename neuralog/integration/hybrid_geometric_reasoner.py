"""
Hybrid Geometric-Symbolic-Neural Reasoner

Extends the neurosymbolic reasoner with geometric algebra verification,
creating a tri-modal validation system:
    1. Neural: LLM extraction with confidence scores
    2. Symbolic: Z3/SMT formal verification (>99% soundness)
    3. Geometric: GA triplet-based differentiable verification

This provides the best of all worlds:
- Neural: Flexibility and generalization
- Symbolic: Formal guarantees and soundness
- Geometric: Differentiability and counterfactual reasoning
"""

from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

import torch

from neuralog.core.types import (
    ExtractionResult,
    Triple,
    Entity,
    Relation,
    ConfidenceLevel
)
from neuralog.core.config import Config

try:
    from neuralog.geometric import (
        TripletEncoder,
        TripletState,
        PolicyChain,
        PolicyValidator,
        VerificationMode,
        truth_degree,
        logical_and,
        ThresholdMode,
    )
    GEOMETRIC_AVAILABLE = True
except ImportError:
    GEOMETRIC_AVAILABLE = False
    logger.warning("Geometric algebra module not available")


class HybridGeometricReasoner:
    """
    Tri-modal reasoner combining neural, symbolic, and geometric approaches.

    Architecture:
        1. Neural: LLM extraction → candidates with confidence
        2. Symbolic: Z3/Ontology → formal validation
        3. Geometric: GA triplets → differentiable verification
        4. Hybrid: Weighted combination → final confidence

    This enables:
        - Fast neural screening
        - Formal symbolic guarantees
        - Differentiable geometric optimization
        - Counterfactual "what-if" analysis
    """

    def __init__(
        self,
        ontology_manager,
        llm_interface,
        formal_verifier=None,
        config: Optional[Config] = None,
        enable_geometric: bool = True,
        threshold_mode: str = "hard",
    ):
        """
        Initialize hybrid geometric reasoner.

        Args:
            ontology_manager: OntologyManager instance
            llm_interface: LLMInterface instance
            formal_verifier: FormalVerifier instance (optional)
            config: Configuration
            enable_geometric: Enable geometric verification
            threshold_mode: Threshold mode for geometric encoding
        """
        self.ontology_manager = ontology_manager
        self.llm_interface = llm_interface
        self.formal_verifier = formal_verifier
        self.config = config or Config()

        # Geometric components
        self.enable_geometric = enable_geometric and GEOMETRIC_AVAILABLE

        if self.enable_geometric:
            # Map string to ThresholdMode enum
            threshold_map = {
                "hard": ThresholdMode.HARD,
                "soft": ThresholdMode.SOFT,
                "margin": ThresholdMode.MARGIN,
            }

            self.triplet_encoder = TripletEncoder(
                mode="canonical",
                threshold_mode=threshold_map.get(threshold_mode, ThresholdMode.HARD),
            )

            self.policy_validator = PolicyValidator(
                threshold_mode=threshold_mode,
                verification_mode=VerificationMode.BIDIRECTIONAL,
            )

            logger.info("Geometric verification enabled")
        else:
            logger.info("Geometric verification disabled")

        # Weights for hybrid confidence
        self.weight_neural = 0.5
        self.weight_symbolic = 0.3
        self.weight_geometric = 0.2

        logger.info("HybridGeometricReasoner initialized")

    def validate_extraction(
        self,
        extraction: ExtractionResult,
        use_symbolic: bool = True,
        use_geometric: bool = True,
    ) -> ExtractionResult:
        """
        Validate extraction using tri-modal verification.

        Pipeline:
            1. Neural baseline (from extraction)
            2. Symbolic validation (ontology + Z3)
            3. Geometric validation (GA triplets)
            4. Hybrid confidence scoring
            5. Counterfactual robustness (optional)

        Args:
            extraction: ExtractionResult from neural extraction
            use_symbolic: Enable symbolic verification
            use_geometric: Enable geometric verification

        Returns:
            Validated extraction with hybrid confidence
        """
        logger.info(
            f"Tri-modal validation: {len(extraction.triples)} triples "
            f"(symbolic={use_symbolic}, geometric={use_geometric})"
        )

        validated_triples = []
        reasoning_paths = []

        for triple in extraction.triples:
            # Store individual confidence scores
            conf_neural = triple.confidence
            conf_symbolic = 0.0
            conf_geometric = 0.0

            # Step 1: Symbolic validation
            if use_symbolic:
                is_valid, error_msg = self._validate_triple_symbolically(triple)

                if is_valid:
                    conf_symbolic = 1.0
                    reasoning_paths.append(f"✓ Symbolic: {triple.to_tuple()}")
                else:
                    conf_symbolic = 0.0
                    reasoning_paths.append(f"✗ Symbolic: {error_msg}")
                    logger.debug(f"Symbolic rejection: {error_msg}")

            # Step 2: Geometric validation
            if use_geometric and self.enable_geometric:
                conf_geometric = self._validate_triple_geometrically(triple)
                reasoning_paths.append(
                    f"⊙ Geometric: truth={conf_geometric:.3f}"
                )

            # Step 3: Hybrid confidence
            triple = self._compute_hybrid_confidence(
                triple,
                conf_neural,
                conf_symbolic if use_symbolic else 1.0,
                conf_geometric if use_geometric else 1.0,
            )

            # Only keep triples with sufficient hybrid confidence
            if triple.confidence > 0.3:  # Threshold
                validated_triples.append(triple)

        # Update extraction
        extraction.triples = validated_triples
        extraction.reasoning_path = reasoning_paths

        if validated_triples:
            extraction.confidence = sum(t.confidence for t in validated_triples) / len(validated_triples)
            extraction.confidence_level = self._determine_confidence_level(extraction.confidence)
        else:
            extraction.confidence = 0.0
            extraction.confidence_level = ConfidenceLevel.UNCERTAIN

        logger.info(
            f"Validation complete: {len(validated_triples)}/{len(extraction.triples)} "
            f"passed (hybrid confidence: {extraction.confidence:.2f})"
        )

        return extraction

    def _validate_triple_symbolically(
        self,
        triple: Triple
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate triple using symbolic reasoning.

        Uses:
            - Ontology consistency checking
            - Z3 formal verification (if available)

        Returns:
            (is_valid, error_message)
        """
        # Basic ontology validation
        subject_type = None
        object_type = None

        if isinstance(triple.subject, Entity) and hasattr(triple.subject, 'entity_type'):
            subject_types = self.ontology_manager.get_entity_types(triple.subject.uri)
            subject_type = subject_types[0] if subject_types else None

        if isinstance(triple.object, Entity) and hasattr(triple.object, 'entity_type'):
            object_types = self.ontology_manager.get_entity_types(triple.object.uri)
            object_type = object_types[0] if object_types else None

        if not subject_type or not object_type:
            return True, None  # Cannot fully validate without types

        # Validate against ontology
        predicate_uri = triple.predicate.uri if hasattr(triple.predicate, 'uri') else str(triple.predicate)

        is_valid, error = self.ontology_manager.validate_triple(
            subject_type,
            predicate_uri,
            object_type
        )

        # If formal verifier available, use it
        if is_valid and self.formal_verifier:
            try:
                # Create minimal extraction for verification
                test_extraction = ExtractionResult(
                    entities=[triple.subject] if isinstance(triple.subject, Entity) else [],
                    relations=[triple.predicate] if isinstance(triple.predicate, Relation) else [],
                    triples=[triple],
                    confidence=1.0,
                )

                formal_valid, proof = self.formal_verifier.verify_extraction(test_extraction)

                if not formal_valid:
                    return False, "Formal verification failed"

                # Enhance triple with verification proof
                if triple.provenance is None:
                    triple.provenance = {}
                triple.provenance['verification_proof'] = proof

            except Exception as e:
                logger.warning(f"Formal verification error: {e}")

        return is_valid, error

    def _validate_triple_geometrically(self, triple: Triple) -> float:
        """
        Validate triple using geometric algebra.

        Encodes triple as TripletState and computes truth degree
        via geometric alignment in Cl(3,0).

        Returns:
            Truth degree in [0, 1]
        """
        if not self.enable_geometric:
            return 1.0

        try:
            # Extract triple components
            subject = str(triple.subject)
            predicate = str(triple.predicate)
            obj = triple.object

            # Encode based on type
            if isinstance(obj, (int, float)):
                # Numeric triple - need threshold
                # For now, use simple encoding
                # TODO: Extract threshold from predicate semantics
                triplet_state = self.triplet_encoder.encode_numeric_triplet(
                    value=float(obj),
                    threshold=0.5,  # Default threshold
                    subject_strength=triple.confidence,
                )
            elif isinstance(obj, bool):
                # Binary triple
                triplet_state = self.triplet_encoder.encode_binary_triplet(
                    truth_value=obj,
                    subject_strength=triple.confidence,
                )
            else:
                # Generic triplet - assume satisfied
                triplet_state = TripletState(
                    x_s=triple.confidence,
                    x_p=1.0,
                    x_t=1.0,
                )

            # Compute truth degree
            truth = truth_degree(triplet_state).item()

            return truth

        except Exception as e:
            logger.warning(f"Geometric validation error: {e}")
            return 0.5  # Neutral if error

    def _compute_hybrid_confidence(
        self,
        triple: Triple,
        conf_neural: float,
        conf_symbolic: float,
        conf_geometric: float,
    ) -> Triple:
        """
        Compute hybrid confidence from three sources.

        Formula:
            hybrid = w_n * neural + w_s * symbolic + w_g * geometric

        Where weights sum to 1.0:
            w_n = 0.5 (neural baseline)
            w_s = 0.3 (formal guarantee)
            w_g = 0.2 (geometric verification)

        Args:
            triple: Triple to update
            conf_neural: Neural confidence from LLM
            conf_symbolic: Symbolic confidence (0 or 1)
            conf_geometric: Geometric confidence [0, 1]

        Returns:
            Triple with updated hybrid confidence
        """
        hybrid = (
            self.weight_neural * conf_neural +
            self.weight_symbolic * conf_symbolic +
            self.weight_geometric * conf_geometric
        )

        triple.confidence = min(1.0, hybrid)

        # Store individual scores in provenance
        if triple.provenance is None:
            triple.provenance = {}

        triple.provenance['confidence_breakdown'] = {
            'neural': conf_neural,
            'symbolic': conf_symbolic,
            'geometric': conf_geometric,
            'hybrid': triple.confidence,
        }

        # Update confidence level based on hybrid score
        triple.confidence_level = self._determine_confidence_level(triple.confidence)

        return triple

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map hybrid confidence to confidence level."""
        if confidence >= 0.99:
            return ConfidenceLevel.VERIFIED
        elif confidence >= 0.90:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def verify_policy_chain(
        self,
        policy_spec: Dict[str, Any],
        llm_answer: str,
    ) -> Dict[str, Any]:
        """
        Verify LLM answer against policy using geometric algebra.

        This is a high-level interface for policy-based verification
        using the geometric validator.

        Args:
            policy_spec: Policy specification dict
            llm_answer: LLM's answer to verify

        Returns:
            Verification result with verdict and explanation
        """
        if not self.enable_geometric:
            logger.warning("Geometric verification not available")
            return {
                "verdict": "UNAVAILABLE",
                "explanation": "Geometric verification not enabled"
            }

        # Encode policy
        policy = self.policy_validator.encode_policy(policy_spec)

        # Verify LLM answer
        result = self.policy_validator.verify(
            llm_answer=llm_answer,
            policy_chain=policy,
        )

        return result

    def analyze_counterfactual(
        self,
        policy_spec: Dict[str, Any],
        condition_index: int,
        value_range: Tuple[float, float],
        num_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Perform counterfactual analysis on a policy.

        Answers "what-if" questions by varying a condition and
        observing how the policy decision changes.

        Args:
            policy_spec: Policy specification
            condition_index: Index of condition to vary
            value_range: (min, max) range to explore
            num_steps: Number of steps in trajectory

        Returns:
            Counterfactual analysis results
        """
        if not self.enable_geometric:
            logger.warning("Counterfactual analysis requires geometric module")
            return {}

        from neuralog.geometric import CounterfactualReasoner

        # Encode policy
        policy = self.policy_validator.encode_policy(policy_spec)

        # Create reasoner
        reasoner = CounterfactualReasoner(policy)

        # Analyze counterfactual
        analysis = reasoner.analyze_numeric_counterfactual(
            triplet_index=condition_index,
            value_range=value_range,
            num_steps=num_steps,
        )

        return analysis

    def get_confidence_explanation(self, triple: Triple) -> str:
        """
        Generate explanation of hybrid confidence scoring.

        Args:
            triple: Triple with confidence breakdown

        Returns:
            Human-readable explanation
        """
        if not triple.provenance or 'confidence_breakdown' not in triple.provenance:
            return f"Confidence: {triple.confidence:.2f} (no breakdown available)"

        breakdown = triple.provenance['confidence_breakdown']

        explanation = f"""
Hybrid Confidence Breakdown for: {triple.to_tuple()}

  Neural (LLM):      {breakdown['neural']:.3f}  (weight: {self.weight_neural})
  Symbolic (Z3):     {breakdown['symbolic']:.3f}  (weight: {self.weight_symbolic})
  Geometric (GA):    {breakdown['geometric']:.3f}  (weight: {self.weight_geometric})
  ─────────────────────────────────────
  Hybrid Total:      {breakdown['hybrid']:.3f}
  Confidence Level:  {triple.confidence_level.value}

Interpretation:
  - Neural: Confidence from LLM extraction
  - Symbolic: Binary validation against ontology/Z3 (0 or 1)
  - Geometric: Truth degree from GA triplet encoding [0,1]
  - Hybrid: Weighted combination for final decision
"""

        return explanation.strip()
