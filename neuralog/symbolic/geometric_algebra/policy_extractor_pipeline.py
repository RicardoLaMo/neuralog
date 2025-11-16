"""Policy-Aware LLM Extraction Pipeline.

Integrates LLM fact extraction with policy verification, enabling:
- Iterative LLM refinement based on policy constraints
- Confidence feedback to LLM via in-context learning
- Counterfactual explanations for policy violations
- Multi-turn validation loops

Pipeline Flow:
  Raw Text → LLM Extract → GA Verify → Counterfactual → LLM Refine → Final Facts
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from neuralog.core.types import ExtractionResult, Triple
from neuralog.symbolic.geometric_algebra import (
    CounterfactualExplanation,
    NeuraLogGASystem,
    SystemConfig,
)


@dataclass
class ExtractionWithVerification:
    """LLM extraction result with policy verification results."""

    raw_text: str
    extraction: ExtractionResult
    verification_results: List[Dict[str, Any]]
    is_policy_compliant: bool
    compliance_score: float  # [0, 1]
    violations: List[str]
    counterfactuals: List[CounterfactualExplanation]
    extraction_round: int


@dataclass
class RefinementFeedback:
    """Feedback for LLM refinement in multi-turn loop."""

    fact_id: str
    violation_description: str
    suggested_correction: Optional[Dict[str, Any]]
    confidence_score: float
    policy_name: str


class PolicyExtractorPipeline:
    """LLM-Integrated Policy-Aware Extraction Pipeline.

    Orchestrates:
    1. LLM-based fact extraction
    2. Policy verification via GA
    3. Counterfactual explanation generation
    4. Multi-turn refinement via feedback

    Supports iterative refinement where the LLM corrects violations
    identified by policy verification.

    Attributes:
        ga_system: NeuraLogGASystem for policy verification
        llm_interface: LLM interface for extraction and refinement
        max_refinement_rounds: Maximum refinement iterations
        compliance_threshold: Minimum compliance score to accept
    """

    def __init__(
        self,
        ga_system: NeuraLogGASystem,
        llm_interface: Optional[Any] = None,
        max_refinement_rounds: int = 3,
        compliance_threshold: float = 0.85,
        enable_counterfactuals: bool = True,
    ):
        """Initialize the extraction pipeline.

        Args:
            ga_system: NeuraLogGASystem instance
            llm_interface: LLM interface (e.g., Claude, GPT-4)
            max_refinement_rounds: Max refinement iterations
            compliance_threshold: Min compliance score
            enable_counterfactuals: Generate counterfactual explanations
        """
        self.ga_system = ga_system
        self.llm_interface = llm_interface
        self.max_refinement_rounds = max_refinement_rounds
        self.compliance_threshold = compliance_threshold
        self.enable_counterfactuals = enable_counterfactuals

    def extract_and_verify(
        self,
        text: str,
        policy_names: Optional[List[str]] = None,
        enable_refinement: bool = True,
    ) -> ExtractionWithVerification:
        """Extract facts from text and verify against policies.

        Args:
            text: Raw text to extract from
            policy_names: Specific policies to verify
            enable_refinement: Enable multi-turn refinement

        Returns:
            ExtractionWithVerification with full verification details
        """
        extraction_round = 0
        current_text = text
        current_extraction = None
        verification_results = []

        for round_num in range(self.max_refinement_rounds):
            extraction_round = round_num
            current_extraction = self._extract_facts(current_text)

            # Convert extraction to fact values
            fact_values = self._extraction_to_fact_values(current_extraction)
            if not fact_values:
                break

            # Verify facts against policies
            batch_results = self.ga_system.verify_facts(
                [fact_values], policy_names, self.enable_counterfactuals
            )

            if not batch_results:
                break

            result = batch_results[0]
            verification_results.append(result)

            # Check compliance
            is_compliant = result.is_consistent
            compliance_score = result.confidence_score

            if is_compliant or compliance_score >= self.compliance_threshold:
                # Policy satisfied, stop refinement
                break

            # Not compliant - refine if enabled
            if not enable_refinement or not self.llm_interface:
                break

            # Generate refinement feedback
            feedback = self._generate_refinement_feedback(
                result, current_extraction, policy_names
            )

            if not feedback:
                break

            # Refine extraction via LLM
            current_text = self._refine_extraction_with_llm(
                current_text, feedback
            )

        # Prepare final result
        is_policy_compliant = (
            verification_results[-1].is_consistent
            if verification_results
            else False
        )
        compliance_score = (
            verification_results[-1].confidence_score
            if verification_results
            else 0.0
        )
        violations = self._extract_violations(
            verification_results, policy_names
        )
        counterfactuals = self._extract_counterfactuals(verification_results)

        return ExtractionWithVerification(
            raw_text=text,
            extraction=current_extraction,
            verification_results=verification_results,
            is_policy_compliant=is_policy_compliant,
            compliance_score=compliance_score,
            violations=violations,
            counterfactuals=counterfactuals,
            extraction_round=extraction_round,
        )

    def batch_extract_and_verify(
        self,
        texts: List[str],
        policy_names: Optional[List[str]] = None,
        enable_refinement: bool = True,
    ) -> List[ExtractionWithVerification]:
        """Process batch of texts with extraction and verification.

        Args:
            texts: List of texts to process
            policy_names: Policies to verify
            enable_refinement: Enable refinement

        Returns:
            List of ExtractionWithVerification results
        """
        results = []
        for text in texts:
            result = self.extract_and_verify(
                text, policy_names, enable_refinement
            )
            results.append(result)
        return results

    def stream_extract_and_verify(
        self,
        text_generator,
        policy_names: Optional[List[str]] = None,
    ):
        """Stream-process texts with extraction and verification.

        Args:
            text_generator: Generator yielding texts
            policy_names: Policies to verify

        Yields:
            ExtractionWithVerification results
        """
        for text in text_generator:
            result = self.extract_and_verify(
                text, policy_names, enable_refinement=False
            )
            yield result

    def _extract_facts(self, text: str) -> ExtractionResult:
        """Extract facts from text using LLM.

        Args:
            text: Raw text

        Returns:
            ExtractionResult with triples, entities, relations
        """
        if self.llm_interface is None:
            # Placeholder: return empty extraction
            from neuralog.core.types import ConfidenceLevel
            return ExtractionResult(
                triples=[],
                entities=[],
                relations=[],
                confidence=0.0,
                confidence_level=ConfidenceLevel.UNCERTAIN,
                provenance={"method": "none"},
            )

        # Call LLM interface to extract
        extraction = self.llm_interface.extract(text)
        return extraction

    def _extraction_to_fact_values(
        self, extraction: ExtractionResult
    ) -> Dict[str, float]:
        """Convert extraction result to fact values for policy evaluation.

        Args:
            extraction: ExtractionResult

        Returns:
            Dict mapping fact names to numeric values
        """
        fact_values = {}

        # Extract numeric properties from triples
        for triple in extraction.triples:
            # Try to extract numeric object values
            try:
                if isinstance(triple.object, (int, float)):
                    # Simple case: numeric object
                    predicate_name = (
                        triple.predicate.label or triple.predicate.uri
                        if hasattr(triple.predicate, "label")
                        else str(triple.predicate)
                    )
                    fact_values[predicate_name] = float(triple.object)
                elif isinstance(triple.object, str):
                    # Try to parse string as number
                    try:
                        fact_values[str(triple.object)] = float(
                            triple.object
                        )
                    except ValueError:
                        pass
            except (AttributeError, TypeError, ValueError):
                pass

        return fact_values

    def _generate_refinement_feedback(
        self,
        verification_result,
        extraction: ExtractionResult,
        policy_names: Optional[List[str]],
    ) -> List[RefinementFeedback]:
        """Generate feedback for LLM refinement.

        Args:
            verification_result: PolicyVerificationResult
            extraction: Current extraction
            policy_names: Policies to check

        Returns:
            List of RefinementFeedback objects
        """
        feedback = []

        # Check each policy
        policies_to_check = (
            policy_names
            if policy_names
            else list(self.ga_system.policies.keys())
        )

        for policy_name in policies_to_check:
            if not verification_result.is_consistent:
                # Find violated conditions
                policy_spec = self.ga_system.policies.get(policy_name)
                if policy_spec:
                    conditions = policy_spec.get("conditions", [])
                    for condition in conditions:
                        subject = condition.get("subject")
                        threshold = condition.get("threshold")
                        current_val = verification_result.extracted_values.get(
                            subject
                        )

                        if (
                            current_val is not None
                            and current_val < threshold
                        ):
                            # Condition not satisfied
                            suggested = threshold + (threshold * 0.1)
                            feedback.append(
                                RefinementFeedback(
                                    fact_id=f"{subject}_violation",
                                    violation_description=f"Value {current_val} does not meet threshold {threshold}",
                                    suggested_correction={
                                        subject: suggested
                                    },
                                    confidence_score=1.0,
                                    policy_name=policy_name,
                                )
                            )

        return feedback

    def _refine_extraction_with_llm(
        self,
        original_text: str,
        feedback: List[RefinementFeedback],
    ) -> str:
        """Use LLM to refine extraction based on feedback.

        Args:
            original_text: Original extraction text
            feedback: Feedback about violations

        Returns:
            Refined text for re-extraction
        """
        if not self.llm_interface:
            return original_text

        # Build refinement prompt
        feedback_strs = [
            f"- {f.violation_description}: suggested {f.suggested_correction}"
            for f in feedback
        ]

        refinement_prompt = f"""
Previous extraction had policy violations:
{chr(10).join(feedback_strs)}

Please revise the extraction to satisfy these constraints while staying true to the source text.
"""

        # Call LLM for refinement
        try:
            refined = self.llm_interface.refine(
                original_text, refinement_prompt
            )
            return refined
        except Exception:
            # If refinement fails, return original
            return original_text

    def _extract_violations(
        self,
        verification_results: List,
        policy_names: Optional[List[str]],
    ) -> List[str]:
        """Extract human-readable violation descriptions.

        Args:
            verification_results: Verification results
            policy_names: Policy names

        Returns:
            List of violation descriptions
        """
        violations = []

        for result in verification_results:
            if not result.is_consistent:
                policies_to_check = (
                    policy_names
                    if policy_names
                    else list(self.ga_system.policies.keys())
                )

                for policy_name in policies_to_check:
                    policy_spec = self.ga_system.policies.get(policy_name)
                    if policy_spec:
                        conditions = policy_spec.get("conditions", [])
                        for condition in conditions:
                            subject = condition.get("subject")
                            threshold = condition.get("threshold")
                            val = result.extracted_values.get(subject)

                            if val is not None and val < threshold:
                                violations.append(
                                    f"{policy_name}: {subject}={val} < threshold={threshold}"
                                )

        return violations

    def _extract_counterfactuals(
        self,
        verification_results: List,
    ) -> List[CounterfactualExplanation]:
        """Extract counterfactual explanations from results.

        Args:
            verification_results: Verification results

        Returns:
            List of counterfactual explanations
        """
        counterfactuals = []

        for result in verification_results:
            if result.counterfactual:
                counterfactuals.append(result.counterfactual)

        return counterfactuals

    def get_compliance_report(
        self, result: ExtractionWithVerification
    ) -> str:
        """Generate human-readable compliance report.

        Args:
            result: ExtractionWithVerification

        Returns:
            Formatted compliance report
        """
        lines = [
            "=" * 60,
            "Policy Compliance Report",
            "=" * 60,
            f"Extraction Rounds: {result.extraction_round + 1}/{self.max_refinement_rounds}",
            f"Policy Compliant: {result.is_policy_compliant}",
            f"Compliance Score: {result.compliance_score:.2%}",
            "",
        ]

        if result.violations:
            lines.append("Violations Found:")
            for violation in result.violations:
                lines.append(f"  - {violation}")
            lines.append("")

        if result.counterfactuals:
            lines.append("Suggested Corrections (Counterfactuals):")
            for cf in result.counterfactuals:
                lines.append(f"  - Adjust: {cf.counterfactual_values}")
                lines.append(f"    Distance: {cf.distance:.3f}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


class ExtractedFact:
    """Wrapper for extracted fact with metadata."""

    def __init__(
        self,
        values: Dict[str, float],
        policy_name: str,
        compliance_score: float,
        is_compliant: bool,
        source_text: str,
    ):
        """Initialize extracted fact.

        Args:
            values: Fact value dict
            policy_name: Associated policy
            compliance_score: Compliance score [0, 1]
            is_compliant: Is compliant with policy
            source_text: Source text
        """
        self.values = values
        self.policy_name = policy_name
        self.compliance_score = compliance_score
        self.is_compliant = is_compliant
        self.source_text = source_text

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExtractedFact(policy={self.policy_name}, "
            f"compliant={self.is_compliant}, "
            f"score={self.compliance_score:.2%})"
        )
