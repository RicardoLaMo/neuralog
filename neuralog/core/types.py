"""Core data types and structures for NeuraLog."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
import uuid


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    CLASS = "class"
    INDIVIDUAL = "individual"
    LITERAL = "literal"
    BLANK_NODE = "blank_node"


class ConfidenceLevel(Enum):
    """Confidence levels for extracted knowledge."""
    VERIFIED = "verified"  # Formally verified (>99% confidence)
    HIGH = "high"          # >90% confidence
    MEDIUM = "medium"      # 70-90% confidence
    LOW = "low"            # <70% confidence
    UNCERTAIN = "uncertain"  # Requires human review


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    uri: str
    label: Optional[str] = None
    entity_type: EntityType = EntityType.INDIVIDUAL
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    source: Optional[str] = None

    def __hash__(self):
        return hash(self.uri)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.uri == other.uri
        return False


@dataclass
class Relation:
    """Represents a relation/property in the knowledge graph."""
    uri: str
    label: Optional[str] = None
    domain: Optional[str] = None
    range: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    is_functional: bool = False
    is_transitive: bool = False
    is_symmetric: bool = False

    def __hash__(self):
        return hash(self.uri)

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self.uri == other.uri
        return False


@dataclass
class Triple:
    """Represents an RDF triple (subject, predicate, object)."""
    subject: Union[Entity, str]
    predicate: Union[Relation, str]
    object: Union[Entity, str, Any]
    confidence: float = 1.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    provenance: Optional[Dict[str, Any]] = None
    verification_proof: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to simple (s, p, o) tuple."""
        s = self.subject.uri if isinstance(self.subject, Entity) else str(self.subject)
        p = self.predicate.uri if isinstance(self.predicate, Relation) else str(self.predicate)
        o = self.object.uri if isinstance(self.object, Entity) else str(self.object)
        return (s, p, o)

    def __hash__(self):
        return hash(self.to_tuple())


@dataclass
class KnowledgeGraph:
    """Represents a knowledge graph with entities, relations, and triples."""
    name: str
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    triples: List[Triple] = field(default_factory=list)
    ontology_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge graph."""
        self.entities[entity.uri] = entity

    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the knowledge graph."""
        self.relations[relation.uri] = relation

    def add_triple(self, triple: Triple) -> None:
        """Add a triple to the knowledge graph."""
        # Ensure entities and relations are registered
        if isinstance(triple.subject, Entity):
            self.add_entity(triple.subject)
        if isinstance(triple.predicate, Relation):
            self.add_relation(triple.predicate)
        if isinstance(triple.object, Entity):
            self.add_entity(triple.object)

        self.triples.append(triple)

    def get_triples_by_subject(self, subject_uri: str) -> List[Triple]:
        """Get all triples with the given subject."""
        return [
            t for t in self.triples
            if (isinstance(t.subject, Entity) and t.subject.uri == subject_uri)
            or (isinstance(t.subject, str) and t.subject == subject_uri)
        ]

    def get_triples_by_predicate(self, predicate_uri: str) -> List[Triple]:
        """Get all triples with the given predicate."""
        return [
            t for t in self.triples
            if (isinstance(t.predicate, Relation) and t.predicate.uri == predicate_uri)
            or (isinstance(t.predicate, str) and t.predicate == predicate_uri)
        ]

    def get_neighbors(self, entity_uri: str) -> Set[str]:
        """Get all neighboring entities."""
        neighbors = set()
        for triple in self.triples:
            s_uri = triple.subject.uri if isinstance(triple.subject, Entity) else triple.subject
            o_uri = triple.object.uri if isinstance(triple.object, Entity) else str(triple.object)

            if s_uri == entity_uri:
                neighbors.add(o_uri)
            elif o_uri == entity_uri:
                neighbors.add(s_uri)

        return neighbors

    def __len__(self):
        return len(self.triples)


@dataclass
class ExtractionResult:
    """Result from information extraction."""
    triples: List[Triple]
    entities: List[Entity]
    relations: List[Relation]
    confidence: float
    confidence_level: ConfidenceLevel
    provenance: Dict[str, Any]
    verification_status: Optional[str] = None
    reasoning_path: Optional[List[str]] = None

    def to_knowledge_graph(self, name: str) -> KnowledgeGraph:
        """Convert extraction result to a knowledge graph."""
        kg = KnowledgeGraph(name=name)
        for entity in self.entities:
            kg.add_entity(entity)
        for relation in self.relations:
            kg.add_relation(relation)
        for triple in self.triples:
            kg.add_triple(triple)
        return kg


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    name: str
    template: str
    variables: List[str]
    examples: Optional[List[Dict[str, str]]] = None
    system_message: Optional[str] = None

    def format(self, **kwargs) -> str:
        """Format the template with given variables."""
        return self.template.format(**kwargs)


@dataclass
class OntologyAxiom:
    """Represents an ontology axiom."""
    axiom_type: str  # e.g., SubClassOf, EquivalentClasses, etc.
    components: List[str]
    verbalization: Optional[str] = None
    logical_form: Optional[str] = None
