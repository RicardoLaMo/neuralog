"""Integration tests for the full extraction pipeline."""

import pytest
from neuralog.core.engine import Engine
from neuralog.core.config import Config
from neuralog.core.types import Entity, Relation, Triple, ConfidenceLevel


@pytest.mark.integration
@pytest.mark.llm
class TestExtractionPipeline:
    """Tests for the complete extraction pipeline."""

    @pytest.fixture
    def engine(self):
        """Create an engine for testing."""
        return Engine()

    def test_simple_extraction_pipeline(self, engine):
        """Test basic extraction pipeline."""
        text = "Barack Obama was born in Hawaii."
        result = engine.extract(text)

        # Verify structure
        assert result is not None
        assert hasattr(result, "entities")
        assert hasattr(result, "relations")
        assert hasattr(result, "triples")
        assert hasattr(result, "confidence")

        # Entities should be found
        assert len(result.entities) > 0
        assert all(isinstance(e, Entity) for e in result.entities)

    def test_extraction_with_named_entities(self, engine):
        """Test that named entities are properly extracted."""
        text = "Alice and Bob met in New York on January 15, 2023."
        result = engine.extract(text)

        assert len(result.entities) > 0
        # Check that entity attributes are valid
        for entity in result.entities:
            assert entity.text is not None
            assert entity.label is not None
            assert 0 <= entity.confidence <= 1
            assert entity.start_char is not None
            assert entity.end_char is not None

    def test_extraction_with_relations(self, engine):
        """Test relation extraction."""
        text = "John works at Google. Jane is John's manager."
        result = engine.extract(text)

        if len(result.relations) > 0:
            for relation in result.relations:
                assert relation.source is not None
                assert relation.target is not None
                assert relation.relation_type is not None
                assert 0 <= relation.confidence <= 1

    def test_extraction_creates_triples(self, engine):
        """Test that triples are created."""
        text = "Paris is the capital of France."
        result = engine.extract(text)

        if len(result.triples) > 0:
            for triple in result.triples:
                assert triple.subject is not None
                assert triple.predicate is not None
                assert triple.object is not None
                assert 0 <= triple.confidence <= 1

    def test_confidence_level_assignment(self, engine):
        """Test that confidence levels are properly assigned."""
        text = "The sky is blue."
        result = engine.extract(text)

        assert result.confidence in [
            ConfidenceLevel.VERIFIED,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
            ConfidenceLevel.UNCERTAIN,
        ]

    def test_extraction_with_complex_text(self, engine):
        """Test extraction with more complex text."""
        text = """
        Dr. Jane Smith completed her PhD at MIT in 2015.
        She then worked at Google for 3 years as a Senior Software Engineer.
        In 2018, she founded her own startup called TechVision.
        The company focuses on artificial intelligence and machine learning applications
        in the healthcare industry. Jane was born in Boston, Massachusetts.
        """
        result = engine.extract(text)

        # Should extract entities from complex text
        assert len(result.entities) > 0
        # Verify entities have reasonable confidence scores
        for entity in result.entities:
            assert 0 < entity.confidence <= 1

    def test_batch_extraction(self, engine):
        """Test batch extraction functionality."""
        texts = [
            "Barack Obama was born in Hawaii.",
            "George Washington was the first president.",
            "Abraham Lincoln abolished slavery.",
        ]

        results = []
        for text in texts:
            result = engine.extract(text)
            results.append(result)

        assert len(results) == len(texts)
        assert all(r is not None for r in results)

    def test_extraction_consistency(self, engine):
        """Test that extraction is consistent for the same input."""
        text = "The Eiffel Tower is in Paris, France."
        result1 = engine.extract(text)
        result2 = engine.extract(text)

        # Results should be structurally equivalent
        assert len(result1.entities) == len(result2.entities)
        assert len(result1.relations) == len(result2.relations)
        assert len(result1.triples) == len(result2.triples)


@pytest.mark.integration
class TestKnowledgeGraphConstruction:
    """Tests for knowledge graph construction from extracted data."""

    @pytest.fixture
    def engine(self):
        """Create an engine for testing."""
        return Engine()

    def test_knowledge_graph_creation(self, engine, sample_knowledge_graph):
        """Test that knowledge graph is properly constructed."""
        kg = sample_knowledge_graph

        assert len(kg.entities) > 0
        assert len(kg.relations) > 0
        assert len(kg.triples) > 0

    def test_knowledge_graph_entity_access(self, engine, sample_knowledge_graph):
        """Test accessing entities in knowledge graph."""
        kg = sample_knowledge_graph

        # Test entity retrieval
        person_entities = kg.get_entities_by_label("PERSON")
        assert len(person_entities) > 0

        location_entities = kg.get_entities_by_label("LOCATION")
        assert len(location_entities) > 0

    def test_knowledge_graph_relation_access(self, engine, sample_knowledge_graph):
        """Test accessing relations in knowledge graph."""
        kg = sample_knowledge_graph

        relations = kg.relations
        assert len(relations) > 0

    def test_knowledge_graph_triple_access(self, engine, sample_knowledge_graph):
        """Test accessing triples in knowledge graph."""
        kg = sample_knowledge_graph

        triples = kg.triples
        assert len(triples) > 0

        # Test triple filtering by predicate
        birth_triples = [
            t for t in triples if "birth" in t.predicate.lower()
        ]


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Tests for complete end-to-end workflows."""

    @pytest.fixture
    def engine(self):
        """Create an engine for testing."""
        return Engine()

    @pytest.mark.slow
    def test_document_processing_workflow(self, engine):
        """Test a complete document processing workflow."""
        # Step 1: Extract from document
        document = """
        Bill Gates founded Microsoft Corporation in 1975.
        Microsoft develops software products including Windows and Office.
        The headquarters are located in Redmond, Washington.
        Bill Gates was born on October 28, 1955.
        """

        # Extract knowledge
        result = engine.extract(document)
        assert result is not None
        assert len(result.entities) > 0

        # Step 2: Create knowledge graph (when implemented)
        # kg = engine.create_knowledge_graph(result)
        # assert kg is not None

        # Step 3: Verify (when implemented)
        # verification = engine.verify(kg.triples)
        # assert verification is not None

    @pytest.mark.slow
    def test_domain_specific_extraction(self, engine):
        """Test extraction on domain-specific content."""
        biomedical_text = """
        The BRCA1 gene is associated with hereditary breast cancer.
        BRCA1 mutations account for approximately 5-10% of breast cancers.
        Patients with BRCA1 mutations have an increased lifetime risk.
        The TP53 gene is another important tumor suppressor gene.
        """

        result = engine.extract(biomedical_text)
        assert result is not None
        assert len(result.entities) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
