# NeuraLog

**Neural Symbolic AI for Information Extraction**

A modern framework combining symbolic reasoning with neural learning for reliable, explainable knowledge extraction from unstructured data.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

NeuraLog integrates **five** cutting-edge approaches in neural symbolic AI:

1. **Neurosymbolic Formalization & Verification** (inspired by [arxiv:2511.09008](https://arxiv.org/abs/2511.09008))
   - Formal verification for >99% soundness guarantees
   - Auditable proof artifacts
   - LLM-based formalization with automated theorem proving

2. **Reliable Knowledge Graph Distillation** (inspired by [arxiv:2510.09580](https://arxiv.org/abs/2510.09580))
   - Ontology-consistent KG construction
   - High factual accuracy (FActScore)
   - Scalable extraction from unstructured data

3. **Ontology-Aware Graph Embeddings** (modernized from [walking-rdf-and-owl](https://github.com/bio-ontology-research-group/walking-rdf-and-owl))
   - RDF-aware random walks
   - Multiple embedding strategies (Node2Vec, TransE, etc.)
   - Integration with ontology reasoning

4. **Deep Ontology Engineering** (leveraging [DeepOnto](https://github.com/KRR-Oxford/DeepOnto))
   - OWLAPI integration for ontology processing
   - Axiom verbalization for LLM consumption
   - Pre-built tools for matching and subsumption

5. **Semantic Layer with Graph Dynamics** (**NEW** - replaces chunking!)
   - **Generative Semantic Workspace** ([arxiv:2511.07587](https://arxiv.org/abs/2511.07587))
     - Space-time anchored narrative representations
     - 20% better performance on EpBench
     - 51% more token-efficient than traditional RAG
   - **Episodic Transformer Memory** ([episodic-transformer-memory-ppo](https://github.com/MarcoMeter/episodic-transformer-memory-ppo))
     - TransformerXL with sliding window
     - Long-range context without chunking
   - **Graph Dynamics Tracker**
     - Entity evolution over time
     - Temporal edges and state transitions
     - Narrative structure preservation

## Key Features

- 🧠 **Hybrid Reasoning**: Combines neural predictions with symbolic validation
- 🎯 **High Confidence**: Formal verification for critical extractions (>99% soundness)
- 📚 **Ontology-First**: Domain ontologies drive extraction and validation
- 🔍 **Explainable**: All predictions include provenance and reasoning paths
- 🚀 **Scalable**: Efficient processing of large document collections
- 🔌 **Modular**: Clean interfaces, pluggable components
- 🌐 **Multi-LLM**: Supports OpenAI, Anthropic, and local models via Ollama
- 🆕 **Graph Dynamics**: Tracks entity evolution over time, not static snapshots
- 📖 **Narrative Structure**: Preserves temporal and spatial coherence (no chunking!)
- 🧩 **Episodic Memory**: Transformer-based long-range context

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (Information Extraction, Q&A, Knowledge Discovery)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│              Neural-Symbolic Integration Layer               │
│  • Neurosymbolic Reasoner (hybrid inference)                │
│  • Knowledge Graph Distiller (extraction)                   │
│  • Formal Verifier (>99% soundness)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│            Symbolic + Neural Processing Layers               │
│  • Ontology Manager (DeepOnto-based)                        │
│  • Graph Embedder (RDF walks, Node2Vec)                     │
│  • LLM Interface (ontology-aware prompts)                   │
│  • Embedding Models (BERT, SentenceTransformers)            │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.11 or higher
- Java 11+ (for OWLAPI via DeepOnto)

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/neuralog.git
cd neuralog

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For all features
pip install -e ".[all]"
```

### Configuration

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Or set environment variables:

```bash
export NEURALOG_LLM_API_KEY="your-openai-api-key"
export NEURALOG_LLM_PROVIDER="openai"
export NEURALOG_LLM_MODEL="gpt-4"
```

## Production Deployment

For production deployments with **vLLM on NVIDIA H200 GPUs**, NeuraLog supports high-performance inference with models like DeepSeek R1 (671B), Qwen/QwQ-32B, and others.

### Quick Production Setup

```bash
# Install production dependencies
pip install -e ".[production]"

# Start with recommended model (QwQ-32B on single H200)
python -m neuralog --config configs/production_qwen_qwq.yaml

# Or run vLLM as OpenAI-compatible server
./scripts/start_vllm_server.sh Qwen/QwQ-32B-Preview 1
python -m neuralog --config configs/production_vllm_server.yaml
```

### Supported Models

| Model | Parameters | GPUs | Config File |
|-------|-----------|------|-------------|
| **Qwen/QwQ-32B** ⭐ | 32B | 1x H200 | `production_qwen_qwq.yaml` |
| DeepSeek R1 | 671B | 4x H200 | `production_deepseek_r1.yaml` |
| Qwen2.5-72B | 72B | 2x H200 | Custom |
| Llama-3.1-70B | 70B | 2x H200 | Custom |

⭐ **Recommended** for most users: Best balance of performance and cost

### Key Benefits

- **2-4x faster inference** than standard transformers
- **50% less memory** with PagedAttention
- **Multi-GPU support** for large models
- **Prefix caching** for repeated prompts
- **OpenAI-compatible API** for easy integration

### Example: Financial Compliance

```python
from neuralog import Engine
from neuralog.core.config import Config

# Load production config with vLLM
config = Config.from_yaml("configs/production_qwen_qwq.yaml")
engine = Engine(config)

# Extract from regulatory text
policy_text = """
Section 226.9 requires creditors to provide 45-day advance notice
before increasing interest rates on credit card accounts.
"""

kg = engine.extract_with_semantic_workspace(
    text=policy_text,
    workspace_name="TILA_regulations"
)

# Neurosymbolic reasoning (not cosine similarity!)
# Symbolic rules + graph patterns + temporal logic
```

**📖 Full Guide**: See [docs/PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md) for:
- Hardware requirements and model selection
- Configuration and performance tuning
- Multi-GPU setup for large models
- Troubleshooting and optimization tips

## Quick Start

### Basic Information Extraction

```python
from neuralog import Engine
from neuralog.core.config import Config

# Initialize engine
config = Config()
engine = Engine(config)

# Extract knowledge from text
text = """
Albert Einstein was a theoretical physicist who developed the theory of
relativity. He received the Nobel Prize in Physics in 1921.
"""

result = engine.extract_knowledge(text, verify=False)

# Access results
for triple in result.triples:
    print(f"{triple.subject.label} -> {triple.predicate.uri} -> {triple.object.label}")
    print(f"Confidence: {triple.confidence:.2f}")

# Convert to knowledge graph
kg = result.to_knowledge_graph("einstein_kg")
```

### With Ontology and Verification

```python
from pathlib import Path
from neuralog import Engine
from neuralog.core.config import Config

# Initialize
config = Config()
engine = Engine(config)

# Load domain ontology
engine.load_ontology(Path("ontologies/biomedical.owl"))

# Extract with formal verification
text = """
BRCA1 gene mutations are associated with increased risk of breast cancer.
The BRCA1 protein plays a role in DNA repair.
"""

result = engine.extract_knowledge(
    text=text,
    verify=True,  # Enable formal verification
    confidence_threshold=0.8
)

# Only verified facts (>99% soundness)
verified_triples = [
    t for t in result.triples
    if t.confidence_level.value == "verified"
]

for triple in verified_triples:
    print(triple.verification_proof)  # Auditable proof artifact
```

### Graph Embeddings

```python
# Build embeddings for entities
embeddings = engine.build_embeddings(kg, method="rdf_walk")

# Find similar entities
similar = engine.graph_embedder.find_similar_entities(
    embeddings,
    entity_uri="http://example.org/BRCA1",
    top_k=10
)
```

### Semantic Workspace (NEW - Replaces Chunking!)

```python
# Extract with semantic workspace approach
# Uses graph dynamics and episodic memory instead of chunks
kg = engine.extract_with_semantic_workspace(
    text=narrative_text,
    workspace_name="my_narrative",
    use_episodic_memory=True
)

# Track entity evolution over time
trajectory = engine.graph_dynamics.get_entity_trajectory("entity_uri")
for state in trajectory:
    print(f"Position {state.narrative_position}: {state.properties}")

# Query episodic memory
context = engine.episodic_memory.get_context(
    query="What happened at Stanford?",
    top_k=10
)

# Build temporal graph snapshot
snapshot = engine.graph_dynamics.build_snapshot_graph(narrative_position=5)
```

## Examples

See the `examples/` directory:

- `simple_extraction.py`: Basic usage
- `biomedical_extraction.py`: Domain-specific extraction with ontology
- **`semantic_workspace_demo.py`**: **NEW** - Demonstrates graph dynamics vs. chunking
- **`financial_compliance_demo.ipynb`**: **NEW** - Jupyter notebook showing neurosymbolic compliance analysis

Run examples:

```bash
python examples/simple_extraction.py
python examples/biomedical_extraction.py
python examples/semantic_workspace_demo.py  # NEW: See semantic layer in action!

# Run Jupyter notebook
jupyter notebook examples/financial_compliance_demo.ipynb
```

### Featured: Financial Compliance Demo

The [financial_compliance_demo.ipynb](examples/financial_compliance_demo.ipynb) notebook demonstrates:

1. **Policy Extraction**: Extract rules from TILA Regulation Z (financial regulations)
2. **Complaint Analysis**: Process CFPB customer complaints
3. **Violation Detection**: Neurosymbolic reasoning (NOT cosine similarity!)
   - Symbolic policy rules
   - Graph pattern matching
   - Temporal reasoning (45-day notice periods, etc.)
   - Causal chain inference
4. **Explainable Results**: Full reasoning steps with audit trails

## Use Cases

### 1. Biomedical Information Extraction
- Extract disease-gene relationships from literature
- Validate against biomedical ontologies (GO, MONDO, HPO)
- Generate evidence-backed knowledge graphs

### 2. Financial Document Analysis
- Extract entities and relationships from filings
- Verify against regulatory ontologies
- Ensure >99% accuracy for compliance

### 3. Scientific Literature Mining
- Build domain-specific knowledge graphs
- Link to existing ontologies
- Enable semantic search and reasoning

### 4. Enterprise Knowledge Management
- Extract knowledge from internal documents
- Organize via corporate taxonomies
- Intelligent Q&A with provenance

## Configuration

NeuraLog can be configured via:

1. **YAML file**: `Config.from_yaml("configs/custom.yaml")`
2. **Environment variables**: Prefix with `NEURALOG_`
3. **Python code**: Direct instantiation

See `configs/default.yaml` for all options.

### Key Configuration Options

```yaml
llm:
  provider: "openai"  # openai, anthropic, ollama
  model: "gpt-4"
  temperature: 0.1

ontology:
  reasoner: "elk"  # elk, hermit, pellet
  enable_reasoning: true

verification:
  enable_verification: true
  solver: "z3"
  soundness_threshold: 0.99

embedding:
  model: "sentence-transformers/all-mpnet-base-v2"
  walk_length: 80
  num_walks: 10
```

## Advanced Features

### Custom Ontologies

```python
# Load your domain ontology
engine.load_ontology("path/to/ontology.owl")

# Schema is automatically used for:
# - Type-guided extraction
# - Consistency validation
# - Prompt engineering
```

### Formal Verification Policies

```python
# Add custom verification policies
engine.verifier.add_policy("domain_rules", [
    "∀x. Disease(x) ∧ affects(x, y) → Gene(y) ∨ Protein(y)",
    "∀x,y. interactsWith(x, y) → interactsWith(y, x)"  # Symmetry
])
```

### Query Knowledge Graphs

```python
# Natural language query
results = engine.query(
    "What genes are associated with breast cancer?",
    kg=my_kg,
    use_reasoning=True
)

# SPARQL query
results = engine.query("""
    SELECT ?gene ?disease WHERE {
        ?gene :associatedWith ?disease .
        ?disease rdf:type :Cancer .
    }
""", kg=my_kg)
```

## Development

### Running Tests

```bash
pytest tests/
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ --cov=neuralog  # With coverage
```

### Code Quality

```bash
# Format code
black neuralog/

# Lint
ruff neuralog/

# Type checking
mypy neuralog/
```

## Architecture Details

See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive design documentation.

## Roadmap

- [ ] Complete SPARQL query execution
- [ ] Multi-modal support (images, tables)
- [ ] Active learning for uncertain extractions
- [ ] Temporal reasoning
- [ ] Causal relationship inference
- [ ] Federated knowledge graphs
- [ ] Web UI for visualization

## Contributing

Contributions welcome! Please see `CONTRIBUTING.md` for guidelines.

## Citation

If you use NeuraLog in your research, please cite:

```bibtex
@software{neuralog2025,
  title={NeuraLog: Neural Symbolic AI for Information Extraction},
  author={NeuraLog Contributors},
  year={2025},
  url={https://github.com/yourusername/neuralog}
}
```

Related papers that inspired this work:

- Bayless et al., "A Neurosymbolic Approach to Natural Language Formalization and Verification" (2025)
- Belova et al., "GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs" (2025)

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on the shoulders of giants:

- [DeepOnto](https://github.com/KRR-Oxford/DeepOnto) - Ontology processing
- [walking-rdf-and-owl](https://github.com/bio-ontology-research-group/walking-rdf-and-owl) - RDF embeddings concept
- OWLAPI - OWL ontology handling
- PyTorch, Transformers, and the broader ML/NLP ecosystem

## Support

- Documentation: [docs.neuralog.ai](https://docs.neuralog.ai) (coming soon)
- Issues: [GitHub Issues](https://github.com/yourusername/neuralog/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/neuralog/discussions)

---

Made with ❤️ by the NeuraLog team
