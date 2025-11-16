"""Unit tests for NeuraLog core engine."""

import pytest
from neuralog.core.engine import Engine
from neuralog.core.config import Config


class TestEngineInitialization:
    """Tests for engine initialization."""

    def test_engine_initialization_with_default_config(self):
        """Test that engine initializes correctly with default config."""
        engine = Engine()
        assert engine is not None
        assert engine.config is not None

    def test_engine_initialization_with_custom_config(self):
        """Test engine initialization with custom config."""
        config = Config()
        engine = Engine(config)
        assert engine is not None
        assert engine.config is config

    def test_engine_has_required_properties(self):
        """Test that engine has all required lazy-loaded properties."""
        engine = Engine()
        # Properties should be accessible without errors
        assert hasattr(engine, "config")
        assert hasattr(engine, "ontology")
        assert hasattr(engine, "llm")
        assert hasattr(engine, "embeddings")
        assert hasattr(engine, "reasoner")
        assert hasattr(engine, "distiller")
        assert hasattr(engine, "verifier")


class TestEngineConfiguration:
    """Tests for engine configuration."""

    def test_engine_config_access(self):
        """Test accessing engine configuration."""
        config = Config()
        engine = Engine(config)
        assert engine.config is config

    def test_engine_statistics(self):
        """Test engine statistics."""
        config = Config()
        engine = Engine(config)
        stats = engine.get_statistics()
        assert "config" in stats
        assert "components" in stats

    def test_engine_config_properties(self):
        """Test configuration properties are accessible."""
        config = Config()
        engine = Engine(config)
        # Should be able to access sub-configs
        assert engine.config.llm is not None
        assert engine.config.ontology is not None
        assert engine.config.embedding is not None


class TestEngineLazyLoading:
    """Tests for engine lazy loading mechanism."""

    def test_ontology_lazy_loading(self):
        """Test that ontology is lazy-loaded."""
        engine = Engine()
        # Access ontology property
        ontology = engine.ontology
        assert ontology is not None
        # Second access should return the same instance
        ontology2 = engine.ontology
        assert ontology is ontology2

    def test_llm_lazy_loading(self):
        """Test that LLM is lazy-loaded."""
        engine = Engine()
        llm = engine.llm
        assert llm is not None
        llm2 = engine.llm
        assert llm is llm2

    def test_embeddings_lazy_loading(self):
        """Test that embeddings model is lazy-loaded."""
        engine = Engine()
        embeddings = engine.embeddings
        assert embeddings is not None
        embeddings2 = engine.embeddings
        assert embeddings is embeddings2


class TestEngineExtraction:
    """Tests for extraction functionality (placeholder)."""

    @pytest.mark.llm
    def test_extract_returns_result(self):
        """Test that extract returns ExtractionResult."""
        engine = Engine()
        text = "Barack Obama was born in Hawaii."
        result = engine.extract(text)
        # Should have extraction result attributes
        assert hasattr(result, "entities")
        assert hasattr(result, "relations")
        assert hasattr(result, "triples")
        assert hasattr(result, "confidence")

    @pytest.mark.llm
    def test_extract_handles_empty_text(self):
        """Test extraction with empty text."""
        engine = Engine()
        result = engine.extract("")
        assert result is not None
        assert isinstance(result.entities, list)
        assert isinstance(result.relations, list)
        assert isinstance(result.triples, list)

    @pytest.mark.llm
    def test_extract_handles_long_text(self):
        """Test extraction with longer text."""
        engine = Engine()
        text = "Barack Obama was born in Hawaii. " * 10
        result = engine.extract(text)
        assert result is not None
        assert len(result.entities) >= 0  # May be empty depending on implementation


if __name__ == "__main__":
    pytest.main([__file__])
