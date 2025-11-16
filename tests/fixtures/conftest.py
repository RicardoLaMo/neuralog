"""
Pytest configuration and shared fixtures for NeuraLog tests.
"""

import pytest
from neuralog.core.engine import Engine
from neuralog.core.config import Config
from neuralog.core.types import Entity, Relation, Triple, KnowledgeGraph


@pytest.fixture(scope="session")
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture(scope="session")
def engine(config):
    """Create a test engine."""
    return Engine(config=config)


@pytest.fixture
def sample_text():
    """Provide sample text for extraction tests."""
    return "Barack Obama was born in Hawaii. He was the 44th President of the United States."


@pytest.fixture
def sample_entities():
    """Provide sample entities for testing."""
    return [
        Entity(text="Barack Obama", label="PERSON", confidence=0.95, start_char=0, end_char=12),
        Entity(text="Hawaii", label="LOCATION", confidence=0.90, start_char=30, end_char=36),
        Entity(
            text="United States",
            label="LOCATION",
            confidence=0.95,
            start_char=75,
            end_char=88,
        ),
    ]


@pytest.fixture
def sample_relations(sample_entities):
    """Provide sample relations for testing."""
    return [
        Relation(
            source=sample_entities[0].text,
            target=sample_entities[1].text,
            relation_type="BORN_IN",
            confidence=0.92,
        ),
        Relation(
            source=sample_entities[0].text,
            target=sample_entities[2].text,
            relation_type="PRESIDENT_OF",
            confidence=0.95,
        ),
    ]


@pytest.fixture
def sample_triples(sample_entities, sample_relations):
    """Provide sample triples for testing."""
    return [
        Triple(
            subject=sample_entities[0].text,
            predicate="http://example.org/birthplace",
            object=sample_entities[1].text,
            confidence=0.92,
        ),
        Triple(
            subject=sample_entities[0].text,
            predicate="http://example.org/president_of",
            object=sample_entities[2].text,
            confidence=0.95,
        ),
    ]


@pytest.fixture
def sample_knowledge_graph(sample_entities, sample_relations, sample_triples):
    """Provide a sample knowledge graph for testing."""
    return KnowledgeGraph(
        entities=sample_entities,
        relations=sample_relations,
        triples=sample_triples,
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')")
    config.addinivalue_line(
        "markers", "llm: marks tests requiring LLM calls (deselect with '-m \"not llm\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
