"""
Graph Dynamics Tracker for entity evolution and relationship changes over time.

Instead of static knowledge graphs, this maintains:
- Temporal graphs showing how entities and relations evolve
- Event-driven updates based on semantic events
- State transitions and property changes
- Causal chains and narrative structure
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from loguru import logger
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available")

from neuralog.core.types import Entity, Relation, Triple


@dataclass
class EntityState:
    """
    State of an entity at a specific time.

    Captures entity properties at a point in the narrative.
    """
    entity_uri: str
    timestamp: datetime
    narrative_position: int
    properties: Dict[str, Any]
    location: Optional[str] = None
    state_type: str = "normal"  # normal, transient, final
    confidence: float = 1.0


@dataclass
class StateTransition:
    """
    Transition between entity states.

    Represents changes in entity properties over time.
    """
    entity_uri: str
    from_state: EntityState
    to_state: EntityState
    trigger_event: Optional[str] = None  # Event ID that caused transition
    transition_type: str = "property_change"  # property_change, location_change, state_change
    changed_properties: Set[str] = field(default_factory=set)


@dataclass
class TemporalEdge:
    """
    Edge in temporal graph with validity period.

    Relations can appear, disappear, or change over time.
    """
    subject: str
    predicate: str
    object: str
    valid_from: int  # Narrative position
    valid_to: Optional[int] = None  # None = still valid
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphDynamicsTracker:
    """
    Tracks dynamic evolution of knowledge graph over time.

    Key capabilities:
    1. Entity state tracking across narrative
    2. Relationship evolution (appear, change, disappear)
    3. Causal chains and event sequences
    4. Temporal querying (state at time T)
    5. Trajectory analysis (how did entity evolve)

    This provides graph dynamics instead of static snapshots.
    """

    def __init__(self, name: str = "graph_dynamics"):
        """
        Initialize Graph Dynamics Tracker.

        Args:
            name: Tracker name
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, limited functionality")

        self.name = name

        # Entity tracking
        self.entity_states: Dict[str, List[EntityState]] = defaultdict(list)
        self.state_transitions: List[StateTransition] = []

        # Temporal graph
        self.temporal_edges: List[TemporalEdge] = []
        self.active_edges: Dict[Tuple[str, str, str], TemporalEdge] = {}  # (s,p,o) -> edge

        # Narrative tracking
        self.current_position = 0
        self.event_graph = nx.DiGraph() if NETWORKX_AVAILABLE else None

        # Causality tracking
        self.causal_chains: List[List[str]] = []  # Sequences of related events

        logger.info(f"GraphDynamicsTracker '{name}' initialized")

    def update_entity_state(
        self,
        entity: Entity,
        properties: Dict[str, Any],
        narrative_position: int,
        event_id: Optional[str] = None
    ) -> EntityState:
        """
        Update entity state at a narrative position.

        Tracks entity evolution by recording state at each point
        where properties change.

        Args:
            entity: Entity being updated
            properties: Current properties
            narrative_position: Position in narrative
            event_id: Event causing this state

        Returns:
            New entity state
        """
        # Create new state
        new_state = EntityState(
            entity_uri=entity.uri,
            timestamp=datetime.now(),
            narrative_position=narrative_position,
            properties=properties.copy(),
            confidence=entity.confidence
        )

        # Get previous state
        previous_states = self.entity_states.get(entity.uri, [])
        if previous_states:
            prev_state = previous_states[-1]

            # Detect what changed
            changed = set()
            for key in set(prev_state.properties.keys()) | set(properties.keys()):
                if prev_state.properties.get(key) != properties.get(key):
                    changed.add(key)

            if changed:
                # Create transition
                transition = StateTransition(
                    entity_uri=entity.uri,
                    from_state=prev_state,
                    to_state=new_state,
                    trigger_event=event_id,
                    changed_properties=changed
                )
                self.state_transitions.append(transition)

        # Add state
        self.entity_states[entity.uri].append(new_state)

        logger.debug(f"Updated state for {entity.uri} at position {narrative_position}")
        return new_state

    def add_temporal_edge(
        self,
        triple: Triple,
        narrative_position: int,
        event_id: Optional[str] = None
    ) -> TemporalEdge:
        """
        Add or update temporal edge in graph.

        Edges have validity periods - they appear and may disappear
        as the narrative progresses.

        Args:
            triple: Triple to add
            narrative_position: Current narrative position
            event_id: Event establishing this edge

        Returns:
            Temporal edge
        """
        s = triple.subject.uri if hasattr(triple.subject, 'uri') else str(triple.subject)
        p = triple.predicate.uri if hasattr(triple.predicate, 'uri') else str(triple.predicate)
        o = triple.object.uri if hasattr(triple.object, 'uri') else str(triple.object)

        edge_key = (s, p, o)

        # Check if edge already exists
        if edge_key in self.active_edges:
            # Edge still valid, update
            edge = self.active_edges[edge_key]
            edge.confidence = max(edge.confidence, triple.confidence)
        else:
            # New edge
            edge = TemporalEdge(
                subject=s,
                predicate=p,
                object=o,
                valid_from=narrative_position,
                confidence=triple.confidence,
                metadata={"event_id": event_id} if event_id else {}
            )
            self.temporal_edges.append(edge)
            self.active_edges[edge_key] = edge

            # Add to event graph if available
            if self.event_graph is not None:
                self.event_graph.add_edge(s, o, predicate=p, position=narrative_position)

        logger.debug(f"Added temporal edge: {s} -[{p}]-> {o} at position {narrative_position}")
        return edge

    def invalidate_edge(
        self,
        subject: str,
        predicate: str,
        object: str,
        narrative_position: int
    ):
        """
        Mark an edge as no longer valid.

        Used when relationships end or change.

        Args:
            subject: Subject URI
            predicate: Predicate URI
            object: Object URI
            narrative_position: Position where edge becomes invalid
        """
        edge_key = (subject, predicate, object)

        if edge_key in self.active_edges:
            edge = self.active_edges[edge_key]
            edge.valid_to = narrative_position
            del self.active_edges[edge_key]

            logger.debug(f"Invalidated edge at position {narrative_position}: {edge_key}")

    def get_entity_trajectory(
        self,
        entity_uri: str,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None
    ) -> List[EntityState]:
        """
        Get trajectory of entity states over time.

        Shows how entity evolved through the narrative.

        Args:
            entity_uri: Entity URI
            start_position: Start of trajectory (None = beginning)
            end_position: End of trajectory (None = current)

        Returns:
            List of entity states in temporal order
        """
        states = self.entity_states.get(entity_uri, [])

        if start_position is not None:
            states = [s for s in states if s.narrative_position >= start_position]

        if end_position is not None:
            states = [s for s in states if s.narrative_position <= end_position]

        return states

    def get_state_at_position(
        self,
        entity_uri: str,
        narrative_position: int
    ) -> Optional[EntityState]:
        """
        Get entity state at specific narrative position.

        Enables temporal queries: "What was entity's state at time T?"

        Args:
            entity_uri: Entity URI
            narrative_position: Narrative position

        Returns:
            Entity state at that position (or None)
        """
        states = self.entity_states.get(entity_uri, [])

        # Find state at or before position
        valid_states = [s for s in states if s.narrative_position <= narrative_position]

        if valid_states:
            return valid_states[-1]  # Most recent before position

        return None

    def get_active_edges_at_position(
        self,
        narrative_position: int
    ) -> List[TemporalEdge]:
        """
        Get all edges that were valid at a narrative position.

        Reconstructs graph state at a point in time.

        Args:
            narrative_position: Narrative position

        Returns:
            List of active temporal edges
        """
        active = []

        for edge in self.temporal_edges:
            if edge.valid_from <= narrative_position:
                if edge.valid_to is None or edge.valid_to > narrative_position:
                    active.append(edge)

        return active

    def build_snapshot_graph(
        self,
        narrative_position: int
    ) -> Optional[nx.DiGraph]:
        """
        Build NetworkX graph snapshot at a narrative position.

        Args:
            narrative_position: Position in narrative

        Returns:
            NetworkX directed graph (or None if NetworkX unavailable)
        """
        if not NETWORKX_AVAILABLE:
            return None

        graph = nx.DiGraph()

        # Add edges active at this position
        edges = self.get_active_edges_at_position(narrative_position)

        for edge in edges:
            graph.add_edge(
                edge.subject,
                edge.object,
                predicate=edge.predicate,
                confidence=edge.confidence
            )

        # Add node attributes (entity states)
        for entity_uri in graph.nodes():
            state = self.get_state_at_position(entity_uri, narrative_position)
            if state:
                graph.nodes[entity_uri].update(state.properties)

        return graph

    def detect_causal_chains(
        self,
        min_chain_length: int = 2
    ) -> List[List[str]]:
        """
        Detect causal chains in event sequence.

        Finds sequences of events that are causally related.

        Args:
            min_chain_length: Minimum length of chain

        Returns:
            List of event ID chains
        """
        # TODO: Implement sophisticated causal detection
        # For now, use temporal proximity and entity overlap

        chains = []

        # Use state transitions as evidence of causality
        # Events that trigger state transitions are causally related

        logger.info(f"Detected {len(chains)} causal chains")
        return chains

    def compute_graph_metrics(
        self,
        narrative_position: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute graph metrics at a position.

        Analyzes graph structure and dynamics.

        Args:
            narrative_position: Position to analyze (None = current)

        Returns:
            Dictionary of metrics
        """
        if narrative_position is None:
            narrative_position = self.current_position

        if not NETWORKX_AVAILABLE:
            return {
                "num_entities": len(self.entity_states),
                "num_edges": len([e for e in self.temporal_edges if e.valid_to is None]),
                "num_transitions": len(self.state_transitions)
            }

        # Build snapshot
        graph = self.build_snapshot_graph(narrative_position)

        if graph is None or graph.number_of_nodes() == 0:
            return {}

        metrics = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "num_transitions": len(self.state_transitions),
        }

        # Connectivity metrics
        if graph.number_of_nodes() > 0:
            try:
                metrics["avg_clustering"] = nx.average_clustering(graph.to_undirected())
            except:
                pass

        return metrics

    def get_entity_interactions(
        self,
        entity_uri: str,
        narrative_position: Optional[int] = None
    ) -> Set[str]:
        """
        Get all entities that interact with given entity.

        Args:
            entity_uri: Entity URI
            narrative_position: Position to check (None = all time)

        Returns:
            Set of interacting entity URIs
        """
        interactions = set()

        edges = self.temporal_edges if narrative_position is None else \
                self.get_active_edges_at_position(narrative_position)

        for edge in edges:
            if edge.subject == entity_uri:
                interactions.add(edge.object)
            elif edge.object == entity_uri:
                interactions.add(edge.subject)

        return interactions

    def export_temporal_graph(self) -> Dict[str, Any]:
        """
        Export temporal graph in portable format.

        Returns:
            Dictionary representation of temporal graph
        """
        return {
            "name": self.name,
            "entity_states": {
                uri: [
                    {
                        "position": s.narrative_position,
                        "properties": s.properties,
                        "timestamp": s.timestamp.isoformat()
                    }
                    for s in states
                ]
                for uri, states in self.entity_states.items()
            },
            "temporal_edges": [
                {
                    "subject": e.subject,
                    "predicate": e.predicate,
                    "object": e.object,
                    "valid_from": e.valid_from,
                    "valid_to": e.valid_to,
                    "confidence": e.confidence
                }
                for e in self.temporal_edges
            ],
            "transitions": len(self.state_transitions),
            "current_position": self.current_position
        }
