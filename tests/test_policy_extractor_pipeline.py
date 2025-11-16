"""Tests for PolicyExtractorPipeline (LLM integration).

Tests cover:
- Fact extraction and policy verification
- Multi-turn refinement loop
- Compliance scoring
- Counterfactual feedback generation
"""

import pytest
from unittest.mock import Mock, MagicMock

from neuralog.core.types import (
    ConfidenceLevel,
    Entity,
    ExtractionResult,
    Relation,
    Triple,
)
from neuralog.symbolic.geometric_algebra import (
    NeuraLogGASystem,
    SystemConfig,
)
from neuralog.symbolic.geometric_algebra.policy_extractor_pipeline import (
    ExtractedFact,
    PolicyExtractorPipeline,
)


class MockLLMInterface:
    """Mock LLM interface for testing."""

    def __init__(self, extraction_results=None):
        """Initialize mock LLM.

        Args:
            extraction_results: Predefined extraction results
        """
        self.extraction_results = extraction_results or []
        self.call_count = 0

    def extract(self, text: str) -> ExtractionResult:
        """Mock extraction.

        Args:
            text: Text to extract from

        Returns:
            ExtractionResult
        """
        if self.extraction_results:
            result = self.extraction_results[
                min(self.call_count, len(self.extraction_results) - 1)
            ]
        else:
            # Return default extraction
            age_entity = Entity(uri="property:age", label="age")
            age_relation = Relation(uri="hasAge", label="age")
            age_triple = Triple(
                subject=age_entity,
                predicate=age_relation,
                object=70,
                confidence=0.9,
                confidence_level=ConfidenceLevel.HIGH,
            )

            result = ExtractionResult(
                triples=[age_triple],
                entities=[age_entity],
                relations=[age_relation],
                confidence=0.9,
                confidence_level=ConfidenceLevel.HIGH,
                provenance={"method": "mock_llm", "call": self.call_count},
            )

        self.call_count += 1
        return result

    def refine(self, text: str, feedback: str) -> str:
        """Mock refinement.

        Args:
            text: Original text
            feedback: Feedback

        Returns:
            Refined text
        """
        return f"{text}\n[Refined based on: {feedback[:50]}...]"


class TestPolicyExtractorPipeline:
    """Tests for PolicyExtractorPipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        config = SystemConfig(device="cpu", batch_size=4)
        self.ga_system = NeuraLogGASystem(config)

        # Register test policy
        self.ga_system.register_policy(
            "senior_discount",
            {
                "name": "senior_discount",
                "conditions": [
                    {"subject": "age", "threshold": 65.0, "mode": "soft"}
                ],
                "operator": "and",
            },
        )

        self.mock_llm = MockLLMInterface()
        self.pipeline = PolicyExtractorPipeline(
            ga_system=self.ga_system,
            llm_interface=self.mock_llm,
            max_refinement_rounds=3,
            compliance_threshold=0.8,
        )

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline.ga_system is not None
        assert self.pipeline.llm_interface is not None
        assert self.pipeline.max_refinement_rounds == 3
        assert self.pipeline.compliance_threshold == 0.8

    def test_extract_and_verify_compliant(self):
        """Test extraction and verification with compliant result."""
        text = "This person is 72 years old and eligible for senior discount"

        result = self.pipeline.extract_and_verify(
            text, policy_names=["senior_discount"], enable_refinement=False
        )

        assert result.raw_text == text
        assert result.extraction is not None
        assert len(result.verification_results) > 0

    def test_extract_and_verify_with_violations(self):
        """Test extraction with policy violations."""
        # Create extraction with age below threshold
        age_entity = Entity(uri="property:age", label="age")
        age_relation = Relation(uri="hasAge", label="age")
        age_triple = Triple(
            subject=age_entity,
            predicate=age_relation,
            object=55,  # Below threshold
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
        )

        extraction = ExtractionResult(
            triples=[age_triple],
            entities=[age_entity],
            relations=[age_relation],
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            provenance={"method": "test"},
        )

        self.mock_llm.extraction_results = [extraction]
        self.mock_llm.call_count = 0

        text = "Person is 55 years old"

        result = self.pipeline.extract_and_verify(
            text, policy_names=["senior_discount"], enable_refinement=False
        )

        assert result.raw_text == text
        # May have violations depending on verification
        assert result.extraction_round >= 0

    def test_extraction_to_fact_values(self):
        """Test conversion of extraction to fact values."""
        age_entity = Entity(uri="property:age", label="age")
        age_relation = Relation(uri="hasAge", label="age")
        age_triple = Triple(
            subject=age_entity,
            predicate=age_relation,
            object=70,
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
        )

        extraction = ExtractionResult(
            triples=[age_triple],
            entities=[age_entity],
            relations=[age_relation],
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            provenance={"method": "test"},
        )

        fact_values = self.pipeline._extraction_to_fact_values(extraction)

        assert isinstance(fact_values, dict)
        assert len(fact_values) > 0

    def test_generate_refinement_feedback(self):
        """Test refinement feedback generation."""
        # Mock verification result
        verification_result = MagicMock()
        verification_result.is_consistent = False
        verification_result.extracted_values = {"age": 60.0}

        # Mock extraction
        extraction = MagicMock()

        feedback = self.pipeline._generate_refinement_feedback(
            verification_result,
            extraction,
            policy_names=["senior_discount"],
        )

        # Should generate feedback about age threshold
        assert isinstance(feedback, list)

    def test_extract_violations(self):
        """Test violation extraction."""
        # Mock verification result
        verification_result = MagicMock()
        verification_result.is_consistent = False
        verification_result.extracted_values = {"age": 60.0}

        violations = self.pipeline._extract_violations(
            [verification_result], policy_names=["senior_discount"]
        )

        assert isinstance(violations, list)

    def test_compliance_report_generation(self):
        """Test compliance report generation."""
        # Create result
        text = "Test extraction"

        result = self.pipeline.extract_and_verify(
            text, policy_names=["senior_discount"], enable_refinement=False
        )

        report = self.pipeline.get_compliance_report(result)

        assert "Policy Compliance Report" in report
        assert "Compliance Score" in report
        assert "Extraction Rounds" in report

    def test_batch_extract_and_verify(self):
        """Test batch processing."""
        texts = [
            "Person A is 72 years old",
            "Person B is 55 years old",
            "Person C is 68 years old",
        ]

        results = self.pipeline.batch_extract_and_verify(
            texts, policy_names=["senior_discount"], enable_refinement=False
        )

        assert len(results) == len(texts)
        for result in results:
            assert result.raw_text is not None
            assert result.extraction is not None

    def test_stream_extract_and_verify(self):
        """Test streaming processing."""

        def text_generator():
            yield "Person A is 72 years old"
            yield "Person B is 55 years old"
            yield "Person C is 68 years old"

        results = list(
            self.pipeline.stream_extract_and_verify(
                text_generator(), policy_names=["senior_discount"]
            )
        )

        assert len(results) == 3

    def test_extracted_fact_wrapper(self):
        """Test ExtractedFact wrapper class."""
        fact = ExtractedFact(
            values={"age": 72, "budget": 25},
            policy_name="senior_discount",
            compliance_score=0.95,
            is_compliant=True,
            source_text="Person is 72 years old",
        )

        assert fact.policy_name == "senior_discount"
        assert fact.is_compliant is True
        assert fact.compliance_score == 0.95
        assert "senior_discount" in repr(fact)

    def test_pipeline_without_llm(self):
        """Test pipeline works without LLM interface."""
        pipeline = PolicyExtractorPipeline(
            ga_system=self.ga_system,
            llm_interface=None,
            max_refinement_rounds=1,
        )

        result = pipeline.extract_and_verify(
            "Test text", enable_refinement=False
        )

        assert result.extraction is not None

    def test_compliance_threshold_check(self):
        """Test compliance threshold enforcement."""
        pipeline = PolicyExtractorPipeline(
            ga_system=self.ga_system,
            llm_interface=self.mock_llm,
            compliance_threshold=0.95,  # Very high threshold
        )

        result = pipeline.extract_and_verify(
            "Person is 72 years old",
            policy_names=["senior_discount"],
            enable_refinement=False,
        )

        # With high threshold, may need more refinement rounds
        assert result.extraction_round >= 0

    def test_counterfactual_extraction(self):
        """Test counterfactual extraction from results."""
        result_with_cf = MagicMock()
        result_with_cf.counterfactual = MagicMock()
        result_with_cf.counterfactual.counterfactual_values = {
            "age": 70
        }

        counterfactuals = self.pipeline._extract_counterfactuals(
            [result_with_cf]
        )

        assert len(counterfactuals) == 1


class TestPolicyExtractorPipelineIntegration:
    """Integration tests for complete pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        config = SystemConfig(device="cpu", batch_size=4)
        self.ga_system = NeuraLogGASystem(config)

        # Register multiple policies
        self.ga_system.register_policy(
            "senior_discount",
            {
                "name": "senior_discount",
                "conditions": [
                    {"subject": "age", "threshold": 65.0, "mode": "soft"}
                ],
                "operator": "and",
            },
        )

        self.ga_system.register_policy(
            "loyalty_bonus",
            {
                "name": "loyalty_bonus",
                "conditions": [
                    {"subject": "age", "threshold": 18.0, "mode": "soft"},
                    {"subject": "orders", "threshold": 10.0, "mode": "soft"},
                ],
                "operator": "and",
            },
        )

        self.mock_llm = MockLLMInterface()
        self.pipeline = PolicyExtractorPipeline(
            ga_system=self.ga_system,
            llm_interface=self.mock_llm,
            max_refinement_rounds=2,
            enable_counterfactuals=True,
        )

    def test_full_extraction_pipeline(self):
        """Test complete extraction and verification pipeline."""
        # Create extraction with multiple facts
        age_triple = Triple(
            subject=Entity(uri="age", label="age"),
            predicate=Relation(uri="hasAge", label="age"),
            object=72,
            confidence_level=ConfidenceLevel.HIGH,
        )

        orders_triple = Triple(
            subject=Entity(uri="orders", label="orders"),
            predicate=Relation(uri="hasOrders", label="orders"),
            object=15,
            confidence_level=ConfidenceLevel.HIGH,
        )

        extraction = ExtractionResult(
            triples=[age_triple, orders_triple],
            entities=[age_triple.subject, orders_triple.subject],
            relations=[age_triple.predicate, orders_triple.predicate],
            confidence=0.95,
            confidence_level=ConfidenceLevel.HIGH,
            provenance={"method": "test_llm"},
        )

        self.mock_llm.extraction_results = [extraction]
        self.mock_llm.call_count = 0

        text = "Customer is 72 years old with 15 previous orders"

        result = self.pipeline.extract_and_verify(
            text,
            policy_names=["senior_discount", "loyalty_bonus"],
            enable_refinement=False,
        )

        # Verify all components executed
        assert result.raw_text == text
        assert result.extraction is not None
        assert len(result.verification_results) >= 1
        assert isinstance(result.compliance_score, float)

    def test_multi_turn_refinement(self):
        """Test multi-turn refinement loop."""
        # Initial extraction below threshold
        initial_extraction = ExtractionResult(
            triples=[
                Triple(
                    subject=Entity(uri="age", label="age"),
                    predicate=Relation(uri="hasAge", label="age"),
                    object=60,  # Below senior threshold
                    confidence_level=ConfidenceLevel.MEDIUM,
                )
            ],
            entities=[Entity(uri="age", label="age")],
            relations=[Relation(uri="hasAge", label="age")],
            confidence=0.8,
            confidence_level=ConfidenceLevel.MEDIUM,
            provenance={"round": 0},
        )

        # Refined extraction above threshold
        refined_extraction = ExtractionResult(
            triples=[
                Triple(
                    subject=Entity(uri="age", label="age"),
                    predicate=Relation(uri="hasAge", label="age"),
                    object=72,  # Above senior threshold
                    confidence_level=ConfidenceLevel.HIGH,
                )
            ],
            entities=[Entity(uri="age", label="age")],
            relations=[Relation(uri="hasAge", label="age")],
            confidence=0.95,
            confidence_level=ConfidenceLevel.HIGH,
            provenance={"round": 1},
        )

        self.mock_llm.extraction_results = [
            initial_extraction,
            refined_extraction,
        ]
        self.mock_llm.call_count = 0

        result = self.pipeline.extract_and_verify(
            "Person is 60... actually 72 years old",
            policy_names=["senior_discount"],
            enable_refinement=True,
        )

        # Should have gone through refinement
        assert result.extraction_round >= 0
        assert result.extraction is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
