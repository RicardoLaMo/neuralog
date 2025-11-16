# Contributing to NeuraLog

Thank you for your interest in contributing to NeuraLog! This document provides guidelines for contributing to the project and extending its functionality.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Organization](#code-organization)
- [Contributing Guidelines](#contributing-guidelines)
- [Extension Points](#extension-points)
- [Testing](#testing)
- [Commit Guidelines](#commit-guidelines)

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip and virtualenv (recommended)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RicardoLaMo/neuralog.git
cd neuralog
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode with all dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks (optional but recommended):
```bash
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neuralog

# Run specific test file
pytest tests/unit/test_engine.py

# Run specific test
pytest tests/unit/test_engine.py::test_engine_initialization
```

### Code Quality Tools

```bash
# Format code with Black
black neuralog tests examples

# Lint with Ruff
ruff check neuralog tests examples

# Type check with mypy
mypy neuralog

# Run all checks
make check  # if Makefile provided
```

## Code Organization

### Module Structure

```
neuralog/
├── core/              # Core engine, types, and configuration
│   ├── engine.py      # Main Engine orchestrator
│   ├── config.py      # Configuration classes (Pydantic)
│   ├── types.py       # Data structures
│   └── __init__.py
├── symbolic/          # Ontology and symbolic reasoning
│   ├── ontology_manager.py
│   ├── graph_embedder.py
│   ├── reasoner.py
│   └── __init__.py
├── neural/            # Neural/LLM components
│   ├── llm_interface.py
│   ├── embedding_model.py
│   ├── extractors.py
│   └── __init__.py
├── integration/       # Neural-symbolic integration
│   ├── neurosymbolic_reasoner.py
│   ├── kg_distiller.py
│   ├── formal_verifier.py
│   └── __init__.py
├── utils/             # Utilities
│   ├── logger.py
│   ├── serialization.py
│   └── __init__.py
├── cli/               # Command-line interface
│   ├── __init__.py
│   └── commands.py    # CLI commands
├── extensions/        # Extension points and plugins
│   ├── __init__.py
│   ├── llm_provider.py         # Base class for LLM providers
│   ├── ontology_provider.py    # Base class for ontology loaders
│   └── extractor_plugin.py     # Base class for extractors
└── __init__.py
```

### Design Principles

1. **Modularity**: Each module has a single, well-defined responsibility
2. **Loose Coupling**: Modules communicate through interfaces, not implementation details
3. **Lazy Loading**: Components are initialized only when needed (see Engine.py)
4. **Type Safety**: All code is type-hinted for better IDE support and catching errors early
5. **Configuration as Code**: Use Pydantic for all configuration management
6. **Separation of Concerns**: Business logic is separate from data models

## Contributing Guidelines

### Before You Start

1. Check if an issue already exists for what you want to work on
2. If not, create an issue describing the feature or bug fix
3. Wait for feedback from maintainers before starting significant work

### Coding Standards

- **Language**: Python 3.11+ with full type hints
- **Formatting**: Use Black (line length: 100)
- **Linting**: Pass Ruff checks
- **Type Checking**: Pass mypy strict mode
- **Docstrings**: Use Google-style docstrings for all public functions and classes

Example:
```python
def extract_entities(text: str, min_confidence: float = 0.5) -> list[Entity]:
    """Extract entities from text.

    Args:
        text: Input text to process
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        List of extracted Entity objects

    Raises:
        ValueError: If min_confidence is not in range [0, 1]
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
    # implementation
```

### PR Checklist

Before submitting a PR, ensure:

- [ ] Code follows the style guide (Black, Ruff)
- [ ] Type hints are present and correct
- [ ] Docstrings are added/updated
- [ ] Tests are added for new functionality
- [ ] Tests pass: `pytest`
- [ ] Coverage maintained or improved
- [ ] Commit messages are clear and descriptive
- [ ] No breaking changes to public APIs (or documented in PR)

## Extension Points

NeuraLog is designed to be extended easily. Here are the main extension points:

### 1. Custom LLM Providers

Create a new LLM provider by extending the base interface:

```python
# neuralog/extensions/llm_provider.py
from abc import ABC, abstractmethod
from typing import Optional

class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed text to vector."""
        pass

# Example: myproject/llm_provider_custom.py
from neuralog.extensions import LLMProvider

class CustomLLMProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation for your custom LLM
        pass

    def embed(self, text: str) -> list[float]:
        # Implementation for your custom embeddings
        pass

# Register in config
config = Config(
    llm=LLMConfig(provider="custom", model="my-model")
)
```

### 2. Custom Ontology Loaders

Extend ontology loading capabilities:

```python
# neuralog/extensions/ontology_provider.py
from abc import ABC, abstractmethod

class OntologyProvider(ABC):
    """Base class for ontology providers."""

    @abstractmethod
    def load(self, source: str) -> 'Ontology':
        """Load ontology from source."""
        pass

# Example implementation
from neuralog.extensions import OntologyProvider

class DatabaseOntologyProvider(OntologyProvider):
    def load(self, source: str) -> 'Ontology':
        # Load ontology from database
        pass
```

### 3. Custom Extractors

Implement domain-specific extractors:

```python
# neuralog/extensions/extractor_plugin.py
from abc import ABC, abstractmethod
from neuralog.core.types import Entity, Relation

class ExtractorPlugin(ABC):
    """Base class for custom extractors."""

    @abstractmethod
    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        pass

    @abstractmethod
    def extract_relations(self, text: str, entities: list[Entity]) -> list[Relation]:
        """Extract relations between entities."""
        pass

# Example: Custom biomedical extractor
class BiomedicalExtractor(ExtractorPlugin):
    def extract_entities(self, text: str) -> list[Entity]:
        # Domain-specific entity extraction
        pass

    def extract_relations(self, text: str, entities: list[Entity]) -> list[Relation]:
        # Domain-specific relation extraction
        pass
```

### 4. Custom Verification Strategies

Add new formal verification approaches:

```python
# In neuralog/integration/formal_verifier.py, extend the verifier
class CustomVerifier(FormalVerifier):
    def verify(self, knowledge_graph, policy=None):
        """Implement custom verification logic."""
        pass
```

## Testing

### Test Structure

```
tests/
├── __init__.py
├── fixtures/              # Shared test data and fixtures
│   ├── __init__.py
│   ├── sample_data.py    # Sample ontologies, texts, KGs
│   └── conftest.py       # Pytest configuration
├── unit/                 # Unit tests for individual modules
│   ├── test_engine.py
│   ├── test_ontology.py
│   ├── test_extractors.py
│   └── ...
└── integration/          # Integration tests
    ├── test_full_pipeline.py
    ├── test_neural_symbolic.py
    └── ...
```

### Writing Tests

Example test:
```python
import pytest
from neuralog.core.engine import Engine
from neuralog.core.config import Config

@pytest.fixture
def engine():
    """Create test engine."""
    return Engine(config=Config())

def test_entity_extraction(engine):
    """Test entity extraction functionality."""
    text = "Barack Obama was born in Hawaii."
    result = engine.extract(text)

    assert len(result.entities) > 0
    assert any(e.label == "PERSON" for e in result.entities)
```

### Test Markers

Mark tests with markers for selective running:
```python
@pytest.mark.slow
def test_large_ontology_processing():
    pass

@pytest.mark.gpu
def test_gpu_embeddings():
    pass

@pytest.mark.llm
def test_with_real_llm():
    pass

# Run only unit tests
pytest -m "not slow and not gpu and not llm"
```

## Commit Guidelines

### Commit Message Format

Follow conventional commits format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Build, dependencies, tooling

**Examples:**
```
feat(extraction): add support for custom entity extractors

- Implement ExtractorPlugin base class
- Update Engine to support plugin registration
- Add example custom biomedical extractor

Closes #42
```

```
fix(ontology): handle missing inverse properties correctly

The ontology manager was not properly handling cases where
inverse properties were not explicitly defined.
```

## Questions?

Feel free to:
- Create an issue on GitHub
- Check the [API documentation](API.md)
- Review the [Architecture guide](ARCHITECTURE_DEEP_DIVE.md)
- Look at [examples](examples/)

Happy contributing! 🚀
