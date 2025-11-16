"""
Neurosymbolic reasoner combining neural predictions with symbolic validation.

Key innovation: Adaptive reasoning that switches between neural (fast) and
symbolic (precise) based on confidence levels.
"""

from typing import Any, Dict, List, Optional
from loguru import logger

from neuralog.core.types import (
    ExtractionResult,
    Triple,
    Entity,
    Relation,
    ConfidenceLevel
)
from neuralog.core.config import Config


class NeurosymbolicReasoner:
    """
    Hybrid reasoner combining neural and symbolic approaches.

    Architecture:
    1. Neural prediction provides candidates with confidence scores
    2. Symbolic validation checks consistency with ontology
    3. Hybrid scoring combines both sources
    4. Adaptive strategy chooses reasoning method based on confidence
    """

    def __init__(self, ontology_manager, llm_interface, config: Config):
        """
        Initialize neurosymbolic reasoner.

        Args:
            ontology_manager: OntologyManager instance
            llm_interface: LLMInterface instance
            config: Configuration
        """
        self.ontology_manager = ontology_manager
        self.llm_interface = llm_interface
        self.config = config

        logger.info("NeurosymbolicReasoner initialized")

    def validate_extraction(
        self,
        extraction: ExtractionResult
    ) -> ExtractionResult:
        """
        Validate and enhance extraction results using neurosymbolic reasoning.

        Pipeline:
        1. Check ontological consistency
        2. Infer missing types using reasoning
        3. Filter inconsistent triples
        4. Enhance confidence scores
        5. Add reasoning paths for explainability

        Args:
            extraction: Initial extraction result from neural component

        Returns:
            Validated and enhanced extraction result
        """
        logger.info(f"Validating extraction with {len(extraction.triples)} triples")

        validated_triples = []
        reasoning_paths = []

        for triple in extraction.triples:
            # Step 1: Symbolic validation
            is_valid, error_msg = self._validate_triple_symbolically(triple)

            if not is_valid:
                logger.debug(f"Triple rejected by symbolic validation: {error_msg}")
                triple.confidence *= 0.1  # Severely penalize invalid triples
                triple.confidence_level = ConfidenceLevel.UNCERTAIN
                reasoning_paths.append(f"Rejected: {error_msg}")
                continue

            # Step 2: Type inference and enhancement
            triple = self._enhance_triple_with_reasoning(triple)

            # Step 3: Hybrid confidence scoring
            triple = self._compute_hybrid_confidence(triple)

            validated_triples.append(triple)
            reasoning_paths.append(f"Validated: {triple.to_tuple()}")

        # Update extraction result
        extraction.triples = validated_triples
        extraction.reasoning_path = reasoning_paths

        # Recompute overall confidence
        if validated_triples:
            extraction.confidence = sum(t.confidence for t in validated_triples) / len(validated_triples)
            extraction.confidence_level = self._determine_confidence_level(extraction.confidence)
        else:
            extraction.confidence = 0.0
            extraction.confidence_level = ConfidenceLevel.UNCERTAIN

        logger.info(
            f"Validation complete: {len(validated_triples)}/{len(extraction.triples)} "
            f"triples passed (confidence: {extraction.confidence:.2f})"
        )

        return extraction

    def _validate_triple_symbolically(
        self,
        triple: Triple
    ) -> tuple[bool, Optional[str]]:
        """
        Validate triple against ontology constraints.

        Checks:
        - Domain/range constraints
        - Type consistency
        - Cardinality restrictions
        - Disjointness axioms

        Args:
            triple: Triple to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Extract types
        subject_type = None
        object_type = None

        if isinstance(triple.subject, Entity) and hasattr(triple.subject, 'entity_type'):
            # Get actual ontology type
            subject_types = self.ontology_manager.get_entity_types(triple.subject.uri)
            subject_type = subject_types[0] if subject_types else None

        if isinstance(triple.object, Entity) and hasattr(triple.object, 'entity_type'):
            object_types = self.ontology_manager.get_entity_types(triple.object.uri)
            object_type = object_types[0] if object_types else None

        if not subject_type or not object_type:
            # Cannot fully validate without types
            return True, None

        # Validate against ontology
        predicate_uri = triple.predicate.uri if hasattr(triple.predicate, 'uri') else str(triple.predicate)

        is_valid, error = self.ontology_manager.validate_triple(
            subject_type,
            predicate_uri,
            object_type
        )

        return is_valid, error

    def _enhance_triple_with_reasoning(self, triple: Triple) -> Triple:
        """
        Enhance triple with inferred information from reasoning.

        - Infer entity types
        - Add provenance from reasoning
        - Annotate with ontology axioms used
        """
        # Infer types if not present
        if isinstance(triple.subject, Entity):
            inferred_types = self.ontology_manager.get_entity_types(triple.subject.uri)
            if inferred_types and not triple.subject.attributes.get('inferred_types'):
                triple.subject.attributes['inferred_types'] = inferred_types

        if isinstance(triple.object, Entity):
            inferred_types = self.ontology_manager.get_entity_types(triple.object.uri)
            if inferred_types and not triple.object.attributes.get('inferred_types'):
                triple.object.attributes['inferred_types'] = inferred_types

        return triple

    def _compute_hybrid_confidence(self, triple: Triple) -> Triple:
        """
        Compute hybrid confidence combining neural and symbolic evidence.

        Formula:
        hybrid_confidence = alpha * neural_confidence + beta * symbolic_confidence

        Where:
        - neural_confidence: From LLM extraction
        - symbolic_confidence: From ontology consistency (0 or 1)
        - alpha, beta: Weights (default: 0.7, 0.3)
        """
        neural_confidence = triple.confidence
        symbolic_confidence = 1.0  # Passed validation

        # TODO: Make weights configurable
        alpha = 0.7
        beta = 0.3

        hybrid_confidence = alpha * neural_confidence + beta * symbolic_confidence
        triple.confidence = min(1.0, hybrid_confidence)

        return triple

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map confidence score to confidence level."""
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

    def infer_new_knowledge(
        self,
        triples: List[Triple],
        max_inferences: int = 100
    ) -> List[Triple]:
        """
        Infer new triples using neurosymbolic reasoning.

        Combines:
        - Symbolic: Deductive reasoning (e.g., transitivity, inheritance)
        - Neural: Inductive reasoning via LLM

        Args:
            triples: Input triples
            max_inferences: Maximum number of inferences

        Returns:
            List of inferred triples
        """
        logger.info(f"Inferring new knowledge from {len(triples)} triples")

        inferred = []

        # TODO: Implement symbolic inference rules
        # - Transitivity
        # - Property inheritance
        # - Class subsumption

        # TODO: Implement neural inference
        # - Use LLM to suggest plausible inferences
        # - Validate with symbolic reasoner

        logger.info(f"Inferred {len(inferred)} new triples")
        return inferred

    def enhance_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance query results with neurosymbolic reasoning.

        - Add inferred knowledge
        - Compute confidence scores
        - Provide explanations

        Args:
            results: Query results

        Returns:
            Enhanced results
        """
        # TODO: Implement result enhancement
        return results

    def explain_inference(self, triple: Triple) -> str:
        """
        Generate natural language explanation for an inference.

        Uses ontology verbalization and LLM to create human-readable
        explanations of reasoning paths.

        Args:
            triple: Triple to explain

        Returns:
            Natural language explanation
        """
        # Get reasoning path from triple provenance
        if not triple.provenance or 'reasoning_path' not in triple.provenance:
            return "No reasoning path available"

        reasoning_path = triple.provenance['reasoning_path']

        # Verbalize using LLM
        prompt = f"""Explain the following reasoning path in natural language:

{reasoning_path}

Provide a clear, concise explanation suitable for a non-expert.
"""

        explanation = self.llm_interface.generate(
            prompt=prompt,
            temperature=0.3
        )

        return explanation
