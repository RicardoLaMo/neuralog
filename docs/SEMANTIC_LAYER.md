# Semantic Layer: Graph Dynamics vs. Chunking

## Overview

The Semantic Layer represents a major architectural enhancement to NeuraLog, replacing naive text chunking with intelligent semantic processing based on cutting-edge research:

1. **Generative Semantic Workspace** ([arxiv:2511.07587](https://arxiv.org/abs/2511.07587))
   - Space-time anchored narrative representations
   - Operator + Reconciler architecture
   - 20% better performance on EpBench
   - 51% more token-efficient than traditional RAG

2. **Episodic Transformer Memory** ([episodic-transformer-memory-ppo](https://github.com/MarcoMeter/episodic-transformer-memory-ppo))
   - TransformerXL/GTrXL with sliding window
   - Relative positional encodings
   - Bounded memory for efficiency
   - Long-range context maintenance

## The Problem with Chunking

Traditional RAG and information extraction systems split documents into arbitrary chunks:

```python
# OLD APPROACH: Naive chunking
chunks = text.split('\n\n')  # or fixed-size windows
for chunk in chunks:
    extract_from_chunk(chunk)  # Isolated processing
```

### Problems:

1. **Breaks Narrative Flow**: Chunks don't respect semantic boundaries
2. **Loses Temporal Structure**: Can't track "what happened when"
3. **Misses Entity Evolution**: Doesn't capture how entities change over time
4. **No Context Across Chunks**: Each chunk processed independently
5. **Arbitrary Boundaries**: Chunk splits can separate related information

## The Semantic Layer Solution

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Semantic Layer                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Generative Semantic Workspace              │    │
│  │                                                      │    │
│  │  ┌──────────────┐         ┌──────────────┐        │    │
│  │  │   Operator   │────────>│  Reconciler  │        │    │
│  │  │              │         │              │        │    │
│  │  │ Maps obs to  │         │ Integrates   │        │    │
│  │  │ semantic     │         │ into         │        │    │
│  │  │ structures   │         │ workspace    │        │    │
│  │  └──────────────┘         └──────────────┘        │    │
│  │                                                      │    │
│  │         Maintains: Events, Entities, Narrative      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Episodic Transformer Memory                 │    │
│  │                                                      │    │
│  │  [Memory State 1] [Memory State 2] ... [State N]   │    │
│  │         ↓                                           │    │
│  │  TransformerXL with Relative Pos Encoding          │    │
│  │         ↓                                           │    │
│  │  Contextual Representations                         │    │
│  │                                                      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Graph Dynamics Tracker                      │    │
│  │                                                      │    │
│  │  Entity States ───> State Transitions               │    │
│  │  Temporal Edges ──> Active/Inactive                 │    │
│  │  Causal Chains ───> Event Sequences                 │    │
│  │                                                      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Semantic Segmenter                          │    │
│  │                                                      │    │
│  │  Strategies: Event | Entity | Temporal | Hybrid     │    │
│  │  Output: Coherent semantic segments                 │    │
│  │                                                      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Generative Semantic Workspace (GSW)

The workspace maintains a coherent semantic representation of observations.

**Operator**: Maps observations to semantic structures
- Extracts events (actions, state changes, interactions)
- Identifies entities and their properties
- Captures temporal and spatial relationships

**Reconciler**: Integrates structures into persistent workspace
- Resolves entity coreferences
- Establishes temporal ordering
- Checks spatial and logical coherence
- Merges with existing knowledge

**Example**:

```python
from neuralog.semantic import GenerativeSemanticWorkspace

workspace = GenerativeSemanticWorkspace(
    name="narrative_workspace",
    llm_interface=llm,
    ontology_manager=ontology
)

# Ingest observations (not chunks!)
events = workspace.ingest_observation("""
    Dr. Sarah Chen started at MIT in 2015.
    She moved to Stanford in 2018.
    She became director in 2021.
""")

# Workspace maintains coherent narrative
for event in events:
    print(f"Event {event.narrative_position}: {event.event_type}")
    print(f"  Entities: {[e.label for e in event.entities]}")
    print(f"  Temporal: {event.temporal_context}")
    print(f"  Coherence: {event.coherence_score}")
```

### 2. Episodic Transformer Memory

Maintains bounded context across long documents using transformer architecture.

**Features**:
- Sliding window of recent observations
- TransformerXL with relative positional encodings
- Self-attention over memory window
- Efficient long-range dependencies

**Example**:

```python
from neuralog.semantic import EpisodicMemory

memory = EpisodicMemory(
    embedding_model=embedder,
    memory_window=128,  # Bounded window
    num_heads=4,
    num_layers=2
)

# Add observations
for text_segment in segments:
    memory.add_observation(text_segment)

# Query relevant context
context = memory.get_context(
    query="What happened at Stanford?",
    top_k=10
)
```

### 3. Graph Dynamics Tracker

Tracks how entities and relationships evolve over time.

**Capabilities**:
- Entity state tracking across narrative
- Temporal edges with validity periods
- State transitions and triggers
- Trajectory analysis
- Temporal queries ("state at time T")

**Example**:

```python
from neuralog.semantic import GraphDynamicsTracker

tracker = GraphDynamicsTracker(name="entity_evolution")

# Track entity states
for event in events:
    for entity in event.entities:
        tracker.update_entity_state(
            entity=entity,
            properties=entity.attributes,
            narrative_position=event.narrative_position
        )

# Get entity trajectory
trajectory = tracker.get_entity_trajectory("sarah_chen")
for state in trajectory:
    print(f"Position {state.narrative_position}: {state.properties}")

# Query state at specific time
state_at_time = tracker.get_state_at_position(
    entity_uri="sarah_chen",
    narrative_position=5
)
```

### 4. Semantic Segmenter

Segments text based on semantic coherence, not arbitrary boundaries.

**Strategies**:
- **Event-based**: Segment by actions and events
- **Entity-based**: Segment by entity continuity
- **Temporal-based**: Segment by time markers
- **Topic-based**: Segment by semantic topics
- **Hybrid**: Combine multiple signals

**Example**:

```python
from neuralog.semantic import SemanticSegmenter

segmenter = SemanticSegmenter(
    llm_interface=llm,
    strategy="hybrid"  # Best approach
)

segments = segmenter.segment(text)

for segment in segments:
    print(f"{segment.segment_type}: {segment.text[:100]}...")
    print(f"  Entities: {segment.entities}")
    print(f"  Temporal markers: {segment.temporal_markers}")
```

## Using the Semantic Layer

### Basic Usage

```python
from neuralog import Engine

engine = Engine()

# NEW METHOD: Semantic workspace extraction
kg = engine.extract_with_semantic_workspace(
    text=narrative_text,
    workspace_name="my_narrative",
    use_episodic_memory=True
)

# Access graph dynamics
trajectory = engine.graph_dynamics.get_entity_trajectory(entity_uri)
snapshot = engine.graph_dynamics.build_snapshot_graph(narrative_position=10)

# Query episodic memory
context = engine.episodic_memory.get_context(query="...")
```

### Comparison with Traditional Approach

| Feature | Chunking | Semantic Workspace |
|---------|----------|-------------------|
| **Boundary Detection** | Fixed size or arbitrary | Semantic coherence |
| **Context** | Limited to chunk | Full narrative + episodic memory |
| **Entity Tracking** | Per-chunk only | Across entire narrative |
| **Temporal Structure** | Lost | Preserved |
| **Efficiency** | Baseline | 51% more token-efficient |
| **Accuracy** | Baseline | 20% better on EpBench |

## Performance Benefits

Based on research papers:

1. **51% Token Efficiency** ([arxiv:2511.07587](https://arxiv.org/abs/2511.07587))
   - Semantic workspace reduces query-time context tokens
   - More efficient than traditional RAG

2. **20% Performance Improvement**
   - Better accuracy on EpBench (100k-1M tokens)
   - Improved entity tracking and event sequencing

3. **Coherence**
   - Maintains narrative structure
   - Enforces temporal, spatial, logical consistency

## Implementation Details

### Data Structures

**SemanticEvent**: Space-time anchored event
```python
@dataclass
class SemanticEvent:
    event_id: str
    event_type: str  # action, state_change, interaction
    timestamp: datetime
    entities: List[Entity]
    relations: List[Triple]
    spatial_context: Optional[str]
    temporal_context: Optional[str]
    narrative_position: int
    coherence_score: float
```

**EntityState**: Entity at a point in time
```python
@dataclass
class EntityState:
    entity_uri: str
    timestamp: datetime
    narrative_position: int
    properties: Dict[str, Any]
    location: Optional[str]
    state_type: str
    confidence: float
```

**TemporalEdge**: Relationship with validity period
```python
@dataclass
class TemporalEdge:
    subject: str
    predicate: str
    object: str
    valid_from: int  # narrative position
    valid_to: Optional[int]  # None = still valid
    confidence: float
```

## Use Cases

### 1. Long Documents with Temporal Structure
- Scientific papers (introduction → methods → results)
- Legal documents (case history over time)
- Medical records (patient timeline)

### 2. Narrative Texts
- Biographies and histories
- News article sequences
- Story comprehension

### 3. Entity Evolution Tracking
- Career progressions
- Company histories
- Product development timelines

### 4. Multi-Document Coherence
- Research literature (papers over time)
- News coverage (event progression)
- Document collections with temporal relationships

## Future Enhancements

1. **Multi-Modal Semantic Workspace**
   - Images, tables, diagrams
   - Spatial layouts

2. **Causal Chain Detection**
   - Automatic causality inference
   - Counterfactual reasoning

3. **Hierarchical Workspaces**
   - Nested narratives
   - Multi-scale temporal structure

4. **Collaborative Workspaces**
   - Multi-agent semantic integration
   - Consensus building

5. **Adaptive Memory**
   - Dynamic window sizing
   - Importance-based retention

## References

1. Rajesh, S., Holur, P., Duan, C., Chong, D., & Roychowdhury, V. (2025). "Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces." arXiv:2511.07587.

2. Paischer, F., Adler, T., Patil, V., Bitto-Nemling, A., & Holzleitner, M. et al. (2022). "History Compression via Language Models in Reinforcement Learning." ICML 2022.

3. Bayless, S. et al. (2025). "A Neurosymbolic Approach to Natural Language Formalization and Verification." arXiv:2511.09008.

4. Belova, M., Xiao, J., Tuli, S., & Jha, N. (2025). "GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data." arXiv:2510.09580.
