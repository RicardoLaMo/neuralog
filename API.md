# NeuraLog API Reference

Complete API documentation for NeuraLog's public interfaces.

## Table of Contents

- [Core Module](#core-module)
- [Symbolic Module](#symbolic-module)
- [Neural Module](#neural-module)
- [Integration Module](#integration-module)
- [Utils Module](#utils-module)

---

## Core Module

### Engine

The main orchestrator for the NeuraLog system.

```python
from neuralog.core.engine import Engine
from neuralog.core.config import Config

# Initialize with default config
engine = Engine()

# Or with custom config
config = Config(
    llm=LLMConfig(provider="openai", model="gpt-4"),
    ontology=OntologyConfig(reasoner="elk")
)
engine = Engine(config=config)
```

#### Methods

**`extract(text: str) -> ExtractionResult`**

Extract knowledge (entities, relations, triples) from text.

```python
result = engine.extract("Barack Obama was born in Hawaii.")

print(result.entities)      # List of Entity objects
print(result.relations)     # List of Relation objects
print(result.triples)       # List of Triple objects
print(result.confidence)    # ConfidenceLevel enum
```

**`reason(triples: list[Triple]) -> list[Triple]`** *(NotImplementedError)*

Apply logical reasoning over knowledge graph.

```python
inferred_triples = engine.reason(extracted_triples)
```

**`verify(triples: list[Triple]) -> VerificationResult`** *(NotImplementedError)*

Verify triples against ontology constraints.

```python
verification = engine.verify(triples)
print(verification.is_valid)     # bool
print(verification.violations)   # list of constraint violations
```

**`save_knowledge_graph(kg: KnowledgeGraph, path: str)`** *(NotImplementedError)*

Serialize knowledge graph to file.

```python
engine.save_knowledge_graph(kg, "output.ttl")
```

**`load_knowledge_graph(path: str) -> KnowledgeGraph`** *(NotImplementedError)*

Load knowledge graph from file.

```python
kg = engine.load_knowledge_graph("saved_kg.ttl")
```

#### Properties

**`config: Config`**

Access the engine's configuration.

```python
print(engine.config.llm.model)
print(engine.config.ontology.reasoner)
```

**`ontology: OntologyManager`**

Access the ontology manager (lazy-loaded).

```python
entities = engine.ontology.get_entities()
relations = engine.ontology.get_relations()
```

**`llm: LLMInterface`**

Access the LLM interface (lazy-loaded).

```python
response = engine.llm.generate("What is machine learning?")
```

**`embeddings: EmbeddingModel`**

Access the embedding model (lazy-loaded).

```python
vector = engine.embeddings.embed("sample text")
```

**`reasoner: NeurosymbolicReasoner`**

Access the neural-symbolic reasoner (lazy-loaded).

```python
inference = engine.reasoner.infer(triples)
```

**`distiller: KGDistiller`**

Access the knowledge graph distiller (lazy-loaded).

```python
kg = engine.distiller.distill(text)
```

**`verifier: FormalVerifier`**

Access the formal verifier (lazy-loaded).

```python
result = engine.verifier.verify(triples)
```

---

### Config

Pydantic configuration classes for managing settings.

```python
from neuralog.core.config import (
    Config,
    LLMConfig,
    OntologyConfig,
    EmbeddingConfig,
    VerificationConfig,
    StorageConfig
)

config = Config(
    llm=LLMConfig(
        provider="openai",              # "openai", "anthropic", "ollama"
        model="gpt-4",
        temperature=0.7,
        max_tokens=2048,
        api_key="sk-..."
    ),
    ontology=OntologyConfig(
        path="path/to/ontology.owl",
        reasoner="elk",                 # "elk", "hermit", "pellet"
        consistency_check=True
    ),
    embedding=EmbeddingConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",                  # "cpu", "cuda", "mps"
        batch_size=32
    ),
    verification=VerificationConfig(
        solver="z3",                    # "z3", "lean"
        soundness_target=0.99
    ),
    storage=StorageConfig(
        triple_store="fuseki",
        vector_store="chromadb",
        document_store="milvus"
    )
)
```

#### Configuration from Environment

All configuration values can be set via environment variables:

```bash
export NEURALOG_LLM_PROVIDER="openai"
export NEURALOG_LLM_MODEL="gpt-4"
export NEURALOG_LLM_API_KEY="sk-..."
export NEURALOG_ONTOLOGY_PATH="path/to/ontology.owl"
export NEURALOG_ONTOLOGY_REASONER="elk"
export NEURALOG_EMBEDDING_DEVICE="cuda"
```

---

### Types

Core data structures for knowledge representation.

#### Entity

```python
from neuralog.core.types import Entity

entity = Entity(
    text="Barack Obama",
    label="PERSON",
    start_char=0,
    end_char=12,
    confidence=0.95,
    metadata={"role": "US President"}
)

print(entity.text)           # "Barack Obama"
print(entity.label)          # "PERSON"
print(entity.confidence)     # 0.95
```

#### Relation

```python
from neuralog.core.types import Relation

relation = Relation(
    source="Barack Obama",
    target="Hawaii",
    relation_type="BORN_IN",
    confidence=0.92,
    metadata={"year": 1961}
)

print(relation.source)       # "Barack Obama"
print(relation.relation_type) # "BORN_IN"
```

#### Triple

```python
from neuralog.core.types import Triple

triple = Triple(
    subject="Barack Obama",
    predicate="http://example.org/birthplace",
    object="Hawaii",
    confidence=0.92,
    source="document_1.txt"
)

print(triple.subject)        # "Barack Obama"
print(triple.predicate)      # "http://example.org/birthplace"
```

#### KnowledgeGraph

```python
from neuralog.core.types import KnowledgeGraph

kg = KnowledgeGraph(
    triples=[triple1, triple2, ...],
    entities=[entity1, entity2, ...],
    relations=[relation1, relation2, ...]
)

# Add triples
kg.add_triple(triple)

# Query
entities = kg.get_entities_by_label("PERSON")
triples = kg.get_triples_with_predicate("birthplace")
```

#### ExtractionResult

```python
from neuralog.core.types import ExtractionResult

result = ExtractionResult(
    text="Input text",
    entities=[...],
    relations=[...],
    triples=[...],
    confidence=ConfidenceLevel.HIGH
)

print(result.entities)       # List[Entity]
print(result.relations)      # List[Relation]
print(result.triples)        # List[Triple]
```

#### ConfidenceLevel

```python
from neuralog.core.types import ConfidenceLevel

levels = [
    ConfidenceLevel.VERIFIED,    # Formally verified (>0.99)
    ConfidenceLevel.HIGH,        # High confidence (0.8-0.99)
    ConfidenceLevel.MEDIUM,      # Medium confidence (0.6-0.8)
    ConfidenceLevel.LOW,         # Low confidence (0.3-0.6)
    ConfidenceLevel.UNCERTAIN    # Uncertain (<0.3)
]
```

---

## Symbolic Module

### OntologyManager

Manage ontologies and symbolic reasoning.

```python
from neuralog.symbolic.ontology_manager import OntologyManager

manager = OntologyManager(ontology_path="path/to/ontology.owl")
```

#### Methods

**`load_ontology(path: str) -> None`**

Load an OWL/RDF ontology.

```python
manager.load_ontology("biomedical.owl")
```

**`get_entities() -> list[str]`**

Get all entity classes from the ontology.

```python
entities = manager.get_entities()
# ["Disease", "Gene", "Protein", ...]
```

**`get_relations() -> list[str]`**

Get all object properties from the ontology.

```python
relations = manager.get_relations()
# ["hasGene", "causedBy", "treats", ...]
```

**`is_valid_triple(subject: str, predicate: str, object: str) -> bool`**

Check if a triple is valid according to the ontology.

```python
is_valid = manager.is_valid_triple(
    subject="Disease",
    predicate="hasGene",
    object="Gene"
)
```

**`get_ontology_axioms() -> str`**

Get verbalized ontology axioms for LLM prompting.

```python
axioms = manager.get_ontology_axioms()
print(axioms)  # "Disease is a subclass of MedicalConcept..."
```

---

### GraphEmbedder

Embed RDF/Knowledge graphs into vector space.

```python
from neuralog.symbolic.graph_embedder import GraphEmbedder

embedder = GraphEmbedder(embedding_method="node2vec")
```

#### Methods

**`fit(graph: KnowledgeGraph) -> None`**

Train embedder on a knowledge graph.

```python
embedder.fit(kg)
```

**`embed_entity(entity: str) -> list[float]`**

Get vector embedding for an entity.

```python
vector = embedder.embed_entity("Barack Obama")
# [0.1, 0.2, 0.3, ...]
```

**`embed_relation(relation: str) -> list[float]`**

Get vector embedding for a relation.

```python
vector = embedder.embed_relation("birthplace")
```

**`most_similar(entity: str, top_k: int = 10) -> list[tuple[str, float]]`**

Find most similar entities.

```python
similar = embedder.most_similar("Disease", top_k=5)
# [("Illness", 0.92), ("Condition", 0.89), ...]
```

---

### SymbolicReasoner

Perform symbolic reasoning using SPARQL queries.

```python
from neuralog.symbolic.reasoner import SymbolicReasoner

reasoner = SymbolicReasoner(kg)
```

#### Methods

**`query(sparql_query: str) -> list[dict]`** *(NotImplementedError)*

Execute a SPARQL query.

```python
results = reasoner.query("""
    SELECT ?disease ?gene
    WHERE {
        ?disease <http://example.org/hasGene> ?gene .
    }
""")
```

**`infer_triples(triples: list[Triple]) -> list[Triple]`** *(NotImplementedError)*

Infer new triples from existing ones using ontology rules.

```python
inferred = reasoner.infer_triples(triples)
```

---

## Neural Module

### LLMInterface

Multi-provider LLM abstraction layer.

```python
from neuralog.neural.llm_interface import LLMInterface
from neuralog.core.config import LLMConfig

config = LLMConfig(provider="openai", model="gpt-4")
llm = LLMInterface(config=config)
```

#### Methods

**`generate(prompt: str, **kwargs) -> str`**

Generate text from a prompt.

```python
response = llm.generate(
    "Extract entities from: Barack Obama was born in Hawaii.",
    temperature=0.5,
    max_tokens=100
)
```

**`generate_with_format(prompt: str, response_format: str) -> str`** *(Partial)*

Generate with structured output format (JSON, XML).

```python
response = llm.generate_with_format(
    prompt="Extract as JSON",
    response_format="json"
)
```

**`batch_generate(prompts: list[str]) -> list[str]`**

Generate responses for multiple prompts.

```python
responses = llm.batch_generate([
    "What is AI?",
    "What is ML?",
    "What is DL?"
])
```

**`count_tokens(text: str) -> int`**

Count tokens in text (for cost estimation).

```python
tokens = llm.count_tokens("Hello world")
```

---

### EmbeddingModel

Embed text to vectors using sentence transformers.

```python
from neuralog.neural.embedding_model import EmbeddingModel
from neuralog.core.config import EmbeddingConfig

config = EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2")
embeddings = EmbeddingModel(config=config)
```

#### Methods

**`embed(text: str) -> list[float]`**

Embed a single text.

```python
vector = embeddings.embed("Barack Obama")
# [0.1, 0.2, 0.3, ...]
```

**`embed_batch(texts: list[str]) -> list[list[float]]`**

Embed multiple texts efficiently.

```python
vectors = embeddings.embed_batch(["Obama", "Biden", "Trump"])
```

**`similarity(text1: str, text2: str) -> float`**

Compute cosine similarity between two texts.

```python
sim = embeddings.similarity("Obama", "Barack Obama")
# 0.95
```

---

### EntityExtractor

Extract entities from text. *(Stub - TODO)*

```python
from neuralog.neural.extractors import EntityExtractor

extractor = EntityExtractor()
entities = extractor.extract("Barack Obama was born in Hawaii.")
```

### RelationExtractor

Extract relations between entities. *(Stub - TODO)*

```python
from neuralog.neural.extractors import RelationExtractor

extractor = RelationExtractor()
relations = extractor.extract(text, entities)
```

---

## Integration Module

### NeurosymbolicReasoner

Hybrid neural-symbolic inference engine.

```python
from neuralog.integration.neurosymbolic_reasoner import NeurosymbolicReasoner

reasoner = NeurosymbolicReasoner(
    ontology_manager=engine.ontology,
    llm=engine.llm,
    embedding_model=engine.embeddings
)
```

#### Methods

**`infer(triples: list[Triple]) -> list[Triple]`**

Perform neural-symbolic inference.

```python
inferred = reasoner.infer(triples)
```

**`score_triple(triple: Triple) -> float`**

Compute confidence score for a triple.

```python
score = reasoner.score_triple(triple)
# 0.92
```

---

### KGDistiller

Knowledge Graph distillation pipeline.

```python
from neuralog.integration.kg_distiller import KGDistiller

distiller = KGDistiller(
    ontology=engine.ontology,
    llm=engine.llm,
    embedding_model=engine.embeddings
)
```

#### Methods

**`distill(text: str) -> KnowledgeGraph`**

Extract knowledge graph from text.

```python
kg = distiller.distill("Barack Obama was born in Hawaii.")
```

**`distill_batch(texts: list[str]) -> list[KnowledgeGraph]`**

Distill multiple texts.

```python
graphs = distiller.distill_batch(documents)
```

---

### FormalVerifier

Formal verification using Z3 solver.

```python
from neuralog.integration.formal_verifier import FormalVerifier

verifier = FormalVerifier(solver="z3")
```

#### Methods

**`verify(triples: list[Triple], policy: str = None) -> VerificationResult`**

Verify triples against ontology constraints.

```python
result = verifier.verify(triples)
print(result.is_valid)      # bool
print(result.soundness)     # 0.99
print(result.violations)    # list of violations
```

---

## Utils Module

### Logger

Structured logging with Loguru.

```python
from neuralog.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.debug(f"Entity: {entity}")
logger.warning("Low confidence score")
logger.error("Failed to load ontology")
```

### Serialization

Serialize knowledge graphs to different formats.

```python
from neuralog.utils.serialization import KGSerializer

serializer = KGSerializer()

# To JSON
json_str = serializer.to_json(kg)

# To RDF Turtle
ttl_str = serializer.to_turtle(kg)

# To RDF XML
xml_str = serializer.to_xml(kg)

# From JSON
kg = serializer.from_json(json_str)
```

---

## Usage Examples

See the [examples/](examples/) directory for complete working examples:

- `simple_extraction.py` - Basic knowledge extraction
- `biomedical_extraction.py` - Domain-specific extraction with ontology
- `custom_llm_provider.py` - Using a custom LLM provider
- `batch_processing.py` - Processing multiple documents

---

## Error Handling

Most API calls can raise the following exceptions:

```python
from neuralog.core.errors import (
    NeuraLogError,           # Base exception
    ConfigurationError,      # Invalid configuration
    OntologyError,           # Ontology loading/processing error
    LLMError,                # LLM API error
    ExtractionError,         # Extraction failed
    VerificationError        # Verification failed
)

try:
    result = engine.extract(text)
except ConfigurationError as e:
    logger.error(f"Config error: {e}")
except LLMError as e:
    logger.error(f"LLM error: {e}")
except ExtractionError as e:
    logger.error(f"Extraction error: {e}")
```

---

## Version Information

- **Latest Version**: 0.1.0 (Alpha)
- **Python**: 3.11+
- **License**: Apache 2.0

For updates and migration guides, see [CHANGELOG.md](docs/CHANGELOG.md).
