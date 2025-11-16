# NeuraLog: Neural Symbolic AI Architecture

## Overview

NeuraLog is a modern neural symbolic AI system for information extraction that combines:
- **Symbolic reasoning** via OWL/RDF ontologies and knowledge graphs
- **Neural learning** via LLMs and deep learning embeddings
- **Formal verification** for high-confidence predictions
- **Knowledge graph construction** from unstructured data

## Core Philosophy

Integration of four key approaches:
1. **Neurosymbolic formalization & verification** (inspired by ARc paper - arxiv:2511.09008)
2. **Reliable KG distillation** (inspired by GraphMERT - arxiv:2510.09580)
3. **Ontology-aware graph embeddings** (modernized from walking-rdf-and-owl)
4. **Deep ontology engineering** (leveraging DeepOnto patterns)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (Information Extraction, Q&A, Knowledge Discovery)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                  NeuraLog Core Engine                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ LLM Interface   │  │ Verification     │  │ KG Manager │ │
│  │ - Prompt Eng.   │  │ - Formalization  │  │ - CRUD     │ │
│  │ - Context Mgmt  │  │ - Logic Checking │  │ - Query    │ │
│  └─────────────────┘  └──────────────────┘  └────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│              Neural-Symbolic Integration Layer               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Neurosymbolic Reasoner                              │  │
│  │  - Hybrid inference (neural + symbolic)              │  │
│  │  - Uncertainty quantification                        │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Knowledge Graph Distiller                           │  │
│  │  - Entity/relation extraction from text             │  │
│  │  - Ontology-guided validation                        │  │
│  │  - Triple generation & refinement                    │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                  Symbolic Processing Layer                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ Ontology        │  │ Graph Embeddings │  │ Reasoner   │ │
│  │ Manager         │  │ - RDF Walk       │  │ - OWL EL   │ │
│  │ (DeepOnto-based)│  │ - Node2Vec       │  │ - SWRL     │ │
│  │ - Verbalizer    │  │ - Hyperbolic     │  │ - DL Query │ │
│  │ - Normalizer    │  │ - Contextual     │  │            │ │
│  └─────────────────┘  └──────────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                  Neural Processing Layer                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ Language Models │  │ Embedding Models │  │ Fine-tuned │ │
│  │ - GPT/Claude    │  │ - BERT variants  │  │ Models     │ │
│  │ - Llama         │  │ - SentenceT5     │  │ - Domain   │ │
│  │ - Local LLMs    │  │ - Custom         │  │   specific │ │
│  └─────────────────┘  └──────────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                      Data Layer                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ Triple Store    │  │ Vector Store     │  │ Document   │ │
│  │ - RDF graphs    │  │ - Embeddings     │  │ Store      │ │
│  │ - SPARQL        │  │ - Similarity     │  │ - Raw text │ │
│  └─────────────────┘  └──────────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Ontology Manager (Symbolic Foundation)

**Purpose**: Manage OWL/RDF ontologies for domain knowledge representation

**Features**:
- Ontology loading, validation, and reasoning (via OWLAPI)
- Axiom extraction and normalization
- Verbalization for LLM integration
- Taxonomy construction
- Subsumption hierarchy management

**Technologies**: DeepOnto, OWLAPI, RDFLib

### 2. Graph Embedding Engine

**Purpose**: Generate semantic embeddings from knowledge graphs

**Features**:
- RDF-aware random walks (modernized from walking-rdf-and-owl)
- Multiple embedding strategies:
  - Node2Vec for graph structure
  - TransE/DistMult for relational patterns
  - Hyperbolic embeddings for hierarchies
  - Contextual embeddings (BERT-based)
- Integration with ontology reasoning

**Technologies**: PyTorch Geometric, DGL, NetworkX

### 3. Neurosymbolic Reasoner

**Purpose**: Hybrid inference combining neural and symbolic approaches

**Features**:
- Neural prediction with symbolic constraints
- Confidence scoring with symbolic validation
- Logical consistency checking
- Explainable inference paths
- Uncertainty quantification

**Technologies**: PyTorch, ProbLog, NeuraLogic

### 4. Knowledge Graph Distiller (GraphMERT-inspired)

**Purpose**: Extract reliable, ontology-consistent KGs from unstructured text

**Pipeline**:
1. **Text Analysis**: LLM-based entity and relation extraction
2. **Ontology Mapping**: Align extractions to ontology concepts
3. **Triple Generation**: Create candidate RDF triples
4. **Validation**: Symbolic reasoning for consistency checking
5. **Refinement**: Iterative improvement with feedback
6. **Quality Metrics**: FActScore and ontology compliance

**Technologies**: Transformers, SpaCy, Custom validators

### 5. LLM Integration Layer

**Purpose**: Leverage modern LLMs with prompt engineering

**Features**:
- **Ontology-aware prompting**: Inject schema information
- **Few-shot learning**: Use ontology examples
- **Chain-of-thought**: Guide reasoning with symbolic structure
- **Verbalized axioms**: Convert logic to natural language
- **Context management**: Efficient retrieval and injection

**Prompt Templates**:
- Entity extraction with type constraints
- Relation classification with ontology guidance
- Logical formalization from natural language
- Consistency validation queries

### 6. Formal Verification Module (ARc-inspired)

**Purpose**: Ensure high-confidence predictions with formal guarantees

**Two-Stage Process**:

**Stage 1: Policy Formalization**
- Convert domain knowledge to formal logic
- Human-in-the-loop validation
- Build reusable policy libraries

**Stage 2: Statement Verification**
- LLM generates candidate formalizations
- Automated theorem proving validates correctness
- >99% soundness guarantee
- Produces auditable proof artifacts

**Technologies**: Z3, Lean, Coq, LLM APIs

## Data Flow

### Information Extraction Pipeline

```
Unstructured Text
      ↓
[LLM Entity/Relation Extraction]
      ↓
Candidate Triples + Confidence
      ↓
[Ontology Mapping & Validation]
      ↓
Symbolic Constraints Applied
      ↓
[Neurosymbolic Reasoning]
      ↓
Validated Knowledge Graph
      ↓
[Graph Embedding]
      ↓
Vector Representations
      ↓
[Storage: Triple Store + Vector DB]
```

### Query & Inference Pipeline

```
Natural Language Query
      ↓
[LLM Understanding + Formalization]
      ↓
Structured Query (SPARQL + Embedding Search)
      ↓
[Hybrid Retrieval]
      ├─ Symbolic: SPARQL over KG
      └─ Neural: Vector similarity
      ↓
Candidate Results
      ↓
[Neurosymbolic Reasoning]
      ↓
[Formal Verification (if needed)]
      ↓
Verified Answer + Confidence + Proof
```

## Technical Stack

### Core Languages
- **Python 3.11+**: Primary language
- **Java/Kotlin**: OWLAPI integration (via JPype)
- **Rust**: Performance-critical components (optional)

### Machine Learning
- **PyTorch**: Deep learning framework
- **Transformers**: LLM integration
- **PyTorch Geometric / DGL**: Graph neural networks
- **Sentence-Transformers**: Text embeddings

### Symbolic AI
- **OWLAPI**: OWL ontology processing
- **RDFLib**: RDF graph manipulation
- **ELK / HermiT**: OWL reasoners
- **Z3 / CVC5**: SMT solvers for verification

### Knowledge Graphs
- **Apache Jena / RDFox**: Triple stores
- **Neo4j**: Graph database (optional)
- **SPARQL**: Query language

### Vector Storage
- **Chroma / Qdrant / Weaviate**: Vector databases
- **FAISS**: Similarity search

### LLM Integration
- **OpenAI / Anthropic APIs**: Commercial LLMs
- **Ollama / vLLM**: Local LLM serving
- **LangChain / LlamaIndex**: Orchestration

## Design Principles

### 1. Modularity
- Each component is independently testable
- Clean interfaces between layers
- Plugin architecture for extensibility

### 2. Scalability
- Streaming processing for large documents
- Distributed graph processing support
- Incremental KG updates

### 3. Explainability
- All predictions include provenance
- Symbolic reasoning paths are traceable
- Verification proofs are human-readable

### 4. Reliability
- Formal verification for critical extractions
- Confidence thresholds for automated decisions
- Human-in-the-loop for uncertain cases

### 5. Ontology-First
- Domain ontologies drive the system
- Type constraints guide extraction
- Consistency is enforced symbolically

## Modernization of walking-rdf-and-owl

The legacy approach is updated as follows:

| Legacy (2017) | Modern (2025) |
|---------------|---------------|
| Groovy/Java | Python + JPype for Java interop |
| Custom DeepWalk C++ | PyTorch Geometric / DGL |
| ELK reasoner only | Pluggable reasoners (ELK, HermiT, RDFox) |
| Static embeddings | Contextual + graph embeddings |
| CLI only | REST API + Python SDK |
| No LLM integration | LLM-guided walks and reasoning |

## Integration with DeepOnto

Leverage DeepOnto for:
- Ontology loading and reasoning (`Ontology` class)
- Axiom verbalization for LLM prompts (`OntologyVerbaliser`)
- Subsumption hierarchy extraction (`OntologyTaxonomy`)
- Ontology projection to RDF (`OntologyProjector`)
- Pre-built tools (BERTMap, BERTSubs) for matching and prediction

## Novel Contributions

1. **LLM-Guided Ontology Walking**: Use LLM attention to weight random walks
2. **Verified Information Extraction**: Formal guarantees on critical facts
3. **Ontology-Constrained Generation**: LLMs generate within symbolic bounds
4. **Hybrid Embedding Space**: Combine graph structure, logical axioms, and contextual semantics
5. **Adaptive Reasoning**: Switch between neural (fast) and symbolic (precise) based on confidence

## Use Cases

1. **Biomedical Information Extraction**
   - Extract disease-gene relationships from literature
   - Validate against biomedical ontologies (GO, MONDO, HP)
   - Generate evidence-backed knowledge graphs

2. **Financial Document Analysis**
   - Extract entities and relationships from filings
   - Verify against regulatory ontologies
   - Ensure >99% accuracy for compliance

3. **Scientific Literature Mining**
   - Build domain-specific knowledge graphs
   - Link to existing ontologies (schema.org, domain vocabularies)
   - Enable semantic search and reasoning

4. **Enterprise Knowledge Management**
   - Extract knowledge from internal documents
   - Organize via corporate taxonomies
   - Enable intelligent Q&A with provenance

## Future Enhancements

1. **Multi-modal Support**: Images, tables, diagrams
2. **Active Learning**: Identify uncertain extractions for human review
3. **Federated KGs**: Cross-organization knowledge integration
4. **Temporal Reasoning**: Track knowledge evolution
5. **Causal Reasoning**: Infer causal relationships
6. **Neuro-symbolic Program Synthesis**: Generate extraction programs

## Getting Started

See `docs/getting-started.md` for installation and quick start guide.
