# NeuraLog Extension Guide

This guide explains how to extend and customize NeuraLog for your specific needs.

## Quick Start: Five Types of Extensions

1. **Custom LLM Provider** - Use a different language model
2. **Custom Ontology Loader** - Load ontologies from custom sources
3. **Custom Extractor** - Implement domain-specific extraction
4. **Custom Verification Strategy** - Add custom verification logic
5. **Custom Embedding Method** - Implement alternative graph embeddings

---

## 1. Custom LLM Provider

### When to Use

- You want to use a different LLM provider (e.g., Llama via Ollama, Claude via Anthropic, local model)
- You want to wrap an internal/proprietary LLM service
- You want to add custom preprocessing or post-processing

### Step-by-Step

#### Step 1: Create Extension Module

Create a new file in your project (or `neuralog/extensions/`):

```python
# myproject/custom_llm.py
from abc import ABC, abstractmethod
from typing import Optional, List

class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class CustomLLMProvider(LLMProvider):
    """Example: Custom LLM provider implementation."""

    def __init__(self, api_url: str, api_key: str, model: str = "default"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using custom LLM endpoint."""
        import requests

        response = requests.post(
            f"{self.api_url}/generate",
            json={
                "prompt": prompt,
                "model": self.model,
                "temperature": temperature
            },
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.text}")

        return response.json()["text"]

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts in parallel."""
        return [self.generate(p, **kwargs) for p in prompts]

    def count_tokens(self, text: str) -> int:
        """Estimate token count (simple word-based)."""
        return len(text.split())
```

#### Step 2: Register with Engine

Option A: Via Config

```python
from neuralog.core.engine import Engine
from neuralog.core.config import Config, LLMConfig
from myproject.custom_llm import CustomLLMProvider

# Create provider
llm_provider = CustomLLMProvider(
    api_url="http://localhost:8000",
    api_key="secret-key",
    model="my-model"
)

# Create config (you can still use default config structure)
config = Config()

# Create engine
engine = Engine(config=config)

# Inject custom provider (update engine's llm property)
# Note: This is a workaround; see Option B for cleaner approach
```

Option B: Modify Engine (Recommended for Production)

Edit `neuralog/core/engine.py`:

```python
from myproject.custom_llm import CustomLLMProvider

class Engine:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            if self.config.llm.provider == "custom":
                self._llm = CustomLLMProvider(
                    api_url=self.config.llm.custom_api_url,
                    api_key=self.config.llm.api_key,
                    model=self.config.llm.model
                )
            else:
                # ... existing providers
                pass
        return self._llm
```

Update config:

```python
# .env or config.yaml
NEURALOG_LLM_PROVIDER=custom
NEURALOG_LLM_CUSTOM_API_URL=http://localhost:8000
NEURALOG_LLM_MODEL=my-model
NEURALOG_LLM_API_KEY=secret-key
```

#### Step 3: Use in Pipeline

```python
from neuralog.core.engine import Engine
from neuralog.core.config import Config

engine = Engine()
result = engine.extract("Your text here")

# The custom LLM provider is used internally
print(result.entities)
print(result.relations)
```

### Example: Ollama Local Model

```python
# neuralog/extensions/ollama_provider.py
import requests
from typing import Optional, List

class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
        )
        return response.json()["response"]

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]

    def count_tokens(self, text: str) -> int:
        response = requests.post(
            f"{self.base_url}/api/tokenize",
            json={"model": self.model, "text": text}
        )
        return len(response.json()["tokens"])
```

---

## 2. Custom Ontology Loader

### When to Use

- Load ontologies from databases, REST APIs, or custom formats
- Apply transformations to ontologies before loading
- Support versioned ontologies

### Implementation

```python
# myproject/custom_ontology.py
from abc import ABC, abstractmethod
from rdflib import Graph
from typing import Optional

class OntologyProvider(ABC):
    """Base class for ontology providers."""

    @abstractmethod
    def load(self, source: str) -> Graph:
        """Load ontology from source."""
        pass


class DatabaseOntologyProvider(OntologyProvider):
    """Load ontologies from a database."""

    def __init__(self, db_url: str, db_user: str, db_password: str):
        self.db_url = db_url
        self.db_user = db_user
        self.db_password = db_password

    def load(self, ontology_name: str) -> Graph:
        """Load ontology from database."""
        # Example: query database for RDF/TTL data
        import requests

        response = requests.get(
            f"{self.db_url}/ontologies/{ontology_name}",
            auth=(self.db_user, self.db_password)
        )

        graph = Graph()
        graph.parse(data=response.text, format="turtle")
        return graph


class VersionedOntologyProvider(OntologyProvider):
    """Load specific versions of ontologies."""

    def __init__(self, ontology_dir: str):
        self.ontology_dir = ontology_dir

    def load(self, source: str) -> Graph:
        """
        Load ontology with version.
        Source format: "ontology_name/v1.0"
        """
        parts = source.split("/")
        name = parts[0]
        version = parts[1] if len(parts) > 1 else "latest"

        path = f"{self.ontology_dir}/{name}/{version}/{name}.owl"

        graph = Graph()
        graph.parse(path)
        return graph
```

### Register with Engine

Update `neuralog/core/config.py` and `neuralog/core/engine.py`:

```python
class OntologyConfig(BaseSettings):
    path: str = "default.owl"
    reasoner: str = "elk"
    provider: str = "file"  # New: "file", "database", "versioned"
    db_url: Optional[str] = None  # New
    consistency_check: bool = True

# In engine.py
from myproject.custom_ontology import DatabaseOntologyProvider

@property
def ontology(self):
    if self._ontology is None:
        if self.config.ontology.provider == "database":
            provider = DatabaseOntologyProvider(
                db_url=self.config.ontology.db_url,
                # ... other params
            )
            graph = provider.load(self.config.ontology.path)
        else:
            # ... default file loading
            pass

        self._ontology = OntologyManager(graph=graph)
    return self._ontology
```

---

## 3. Custom Extractors

### When to Use

- Extract domain-specific entities or relations
- Apply custom NLP techniques
- Implement rule-based extraction

### Implementation

```python
# myproject/biomedical_extractor.py
from abc import ABC, abstractmethod
from neuralog.core.types import Entity, Relation
from typing import List

class ExtractorPlugin(ABC):
    """Base class for custom extractors."""

    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        pass

    @abstractmethod
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities."""
        pass


class BiomedicalExtractor(ExtractorPlugin):
    """Domain-specific biomedical extractor."""

    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract biomedical entities using domain knowledge."""
        prompt = f"""
        Extract biomedical entities (DISEASE, GENE, PROTEIN, DRUG) from text:

        Text: {text}

        Format as JSON list:
        [{{"text": "...", "label": "...", "confidence": 0.0}}]
        """

        response = self.llm.generate(prompt)
        # Parse response and create Entity objects
        entities = self._parse_entities(response)
        return entities

    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract biomedical relations."""
        entity_text = ", ".join([e.text for e in entities])

        prompt = f"""
        Given entities: {entity_text}
        Extract biomedical relations from text:

        Text: {text}

        Possible relations: treats, causes, associated_with, regulates
        Format as JSON list.
        """

        response = self.llm.generate(prompt)
        relations = self._parse_relations(response)
        return relations

    def _parse_entities(self, json_response: str) -> List[Entity]:
        import json
        data = json.loads(json_response)
        return [Entity(text=d["text"], label=d["label"], confidence=d["confidence"])
                for d in data]

    def _parse_relations(self, json_response: str) -> List[Relation]:
        import json
        data = json.loads(json_response)
        return [Relation(source=d["source"], target=d["target"],
                        relation_type=d["type"], confidence=d["confidence"])
                for d in data]
```

### Register with Engine

```python
# Update neuralog/core/engine.py
from myproject.biomedical_extractor import BiomedicalExtractor

class Engine:
    def __init__(self, config: Config = None, extractor=None):
        self.config = config or Config()
        self._extractor = extractor

    @property
    def extractor(self):
        if self._extractor is None:
            if self.config.extraction.type == "biomedical":
                self._extractor = BiomedicalExtractor(
                    llm=self.llm,
                    embedding_model=self.embeddings
                )
            else:
                # Default extraction
                from neuralog.integration.kg_distiller import KGDistiller
                self._extractor = KGDistiller(
                    ontology=self.ontology,
                    llm=self.llm,
                    embedding_model=self.embeddings
                )
        return self._extractor
```

### Use Custom Extractor

```python
from neuralog.core.engine import Engine
from myproject.biomedical_extractor import BiomedicalExtractor

engine = Engine()
engine._extractor = BiomedicalExtractor(engine.llm, engine.embeddings)

result = engine.extract("BRCA1 mutations cause breast cancer.")
```

---

## 4. Custom Verification Strategy

### Implementation

```python
# myproject/custom_verifier.py
from neuralog.integration.formal_verifier import FormalVerifier
from neuralog.core.types import Triple
from typing import List

class CustomVerifier(FormalVerifier):
    """Custom verification strategy."""

    def verify(self, triples: List[Triple], policy: str = None) -> dict:
        """Custom verification logic."""

        verified = []
        violations = []

        for triple in triples:
            # Custom verification logic
            if self._is_valid_triple(triple):
                verified.append(triple)
            else:
                violations.append({
                    "triple": triple,
                    "reason": "Custom rule violated"
                })

        return {
            "is_valid": len(violations) == 0,
            "verified": verified,
            "violations": violations,
            "soundness": len(verified) / (len(verified) + len(violations)) if triples else 1.0
        }

    def _is_valid_triple(self, triple: Triple) -> bool:
        # Custom validation logic
        # Example: check prefix, length constraints, etc.
        return True
```

---

## 5. Custom Graph Embedding Method

### Implementation

```python
# myproject/custom_embeddings.py
from neuralog.symbolic.graph_embedder import GraphEmbedder
from neuralog.core.types import KnowledgeGraph
from typing import List

class CustomGraphEmbedder(GraphEmbedder):
    """Custom graph embedding method."""

    def __init__(self, embedding_dim: int = 100, method: str = "custom"):
        super().__init__(embedding_dim=embedding_dim, method=method)

    def fit(self, kg: KnowledgeGraph) -> None:
        """Fit embeddings on knowledge graph."""
        # Implement custom embedding algorithm
        pass

    def embed_entity(self, entity: str) -> List[float]:
        """Embed entity using custom method."""
        # Return embedding vector
        pass
```

---

## Project Structure for Extensions

Recommended structure for your custom extensions:

```
myproject/
├── neuralog_extensions/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── custom_llm.py
│   │   └── ollama.py
│   ├── ontology/
│   │   ├── __init__.py
│   │   └── database_loader.py
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── biomedical.py
│   │   └── financial.py
│   ├── verification/
│   │   ├── __init__.py
│   │   └── custom_verifier.py
│   └── embeddings/
│       ├── __init__.py
│       └── custom_embedder.py
├── configs/
│   └── custom_config.yaml
├── examples/
│   └── use_custom_extensions.py
└── tests/
    └── test_extensions.py
```

---

## Testing Your Extensions

```python
# tests/test_extensions.py
import pytest
from myproject.custom_llm import CustomLLMProvider

@pytest.fixture
def custom_llm():
    return CustomLLMProvider(api_url="http://localhost:8000", api_key="test")

def test_custom_llm_generation(custom_llm):
    result = custom_llm.generate("Hello world")
    assert isinstance(result, str)
    assert len(result) > 0

def test_custom_llm_batch(custom_llm):
    prompts = ["Hello", "World"]
    results = custom_llm.batch_generate(prompts)
    assert len(results) == 2

def test_integration_with_engine():
    from neuralog.core.engine import Engine

    engine = Engine()
    engine._llm = CustomLLMProvider(...)

    result = engine.extract("Test text")
    assert result is not None
```

---

## Best Practices

1. **Type Hints**: Always use type hints in your extensions
2. **Documentation**: Document your custom classes and methods
3. **Error Handling**: Handle errors gracefully and provide meaningful messages
4. **Testing**: Write tests for your custom extensions
5. **Configuration**: Use Pydantic configs for managing settings
6. **Logging**: Use the provided logger for debugging

```python
from neuralog.utils.logger import get_logger

logger = get_logger(__name__)

class MyExtension:
    def process(self, data):
        logger.debug(f"Processing {len(data)} items")
        try:
            result = self._do_processing(data)
            logger.info("Processing completed successfully")
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
```

---

## Publishing Your Extension

To share your extension with the community:

1. Create a GitHub repository: `neuralog-{extension-name}`
2. Use consistent naming: `NeuraLog{ExtensionName}`
3. Include comprehensive documentation
4. Add example code
5. Write tests with >80% coverage
6. Submit to PyPI

---

## Need Help?

- Check [API.md](../API.md) for detailed API reference
- Review [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
- See [examples/](../examples/) for working examples
- Open an issue on GitHub for questions
