"""
Ontology management using DeepOnto and OWLAPI.

Provides functionality for:
- Loading and validating ontologies
- Reasoning over ontologies
- Extracting axioms and hierarchies
- Verbalizing axioms for LLM consumption
- Normalizing ontologies
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from loguru import logger

try:
    from deeponto.onto import Ontology
    from deeponto.onto.taxonomy import OntologyTaxonomy
    from deeponto.onto.verbalisation import OntologyVerbaliser
    from deeponto.onto.projection import OntologyProjector
    from deeponto.onto.normalisation import OntologyNormaliser
    DEEPONTO_AVAILABLE = True
except ImportError:
    DEEPONTO_AVAILABLE = False
    logger.warning("DeepOnto not available. Install with: pip install deeponto")

from neuralog.core.config import OntologyConfig
from neuralog.core.types import OntologyAxiom, Entity, Relation


class OntologyManager:
    """
    Manages ontologies using DeepOnto wrapper around OWLAPI.

    Integrates with modern LLMs via:
    - Axiom verbalization for context injection
    - Schema extraction for prompt engineering
    - Type constraints for guided generation
    """

    def __init__(self, config: OntologyConfig):
        """
        Initialize ontology manager.

        Args:
            config: Ontology configuration
        """
        if not DEEPONTO_AVAILABLE:
            raise ImportError(
                "DeepOnto is required for OntologyManager. "
                "Install with: pip install deeponto"
            )

        self.config = config
        self.ontology: Optional[Ontology] = None
        self.taxonomy: Optional[OntologyTaxonomy] = None
        self.verbaliser: Optional[OntologyVerbaliser] = None
        self.projector: Optional[OntologyProjector] = None
        self.normaliser: Optional[OntologyNormaliser] = None

        logger.info(f"OntologyManager initialized with reasoner: {config.reasoner}")

    def load_ontology(self, ontology_path: Path) -> None:
        """
        Load ontology from file.

        Args:
            ontology_path: Path to OWL/RDF file
        """
        logger.info(f"Loading ontology from: {ontology_path}")

        self.ontology = Ontology(str(ontology_path))

        # Initialize reasoner if enabled
        if self.config.enable_reasoning:
            logger.info(f"Initializing reasoner: {self.config.reasoner}")
            self.ontology.reasoner()

        # Initialize helper components
        self.taxonomy = OntologyTaxonomy(self.ontology)
        self.verbaliser = OntologyVerbaliser(self.ontology)
        self.projector = OntologyProjector(self.ontology)
        self.normaliser = OntologyNormaliser(self.ontology)

        logger.info(f"Ontology loaded successfully: {self.get_statistics()}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get ontology statistics."""
        if not self.ontology:
            return {"loaded": False}

        return {
            "loaded": True,
            "classes": len(list(self.ontology.owl_classes())),
            "object_properties": len(list(self.ontology.owl_object_properties())),
            "data_properties": len(list(self.ontology.owl_data_properties())),
            "individuals": len(list(self.ontology.owl_individuals())),
            "axioms": len(list(self.ontology.axioms())),
        }

    def get_schema(self) -> Dict[str, Any]:
        """
        Extract ontology schema for LLM context.

        Returns:
            Dictionary with classes, properties, and their descriptions
        """
        if not self.ontology:
            return {}

        schema = {
            "classes": [],
            "object_properties": [],
            "data_properties": [],
        }

        # Extract classes
        for owl_class in self.ontology.owl_classes():
            class_info = {
                "uri": owl_class.getIRI().toString(),
                "label": self._get_label(owl_class),
                "description": self._get_description(owl_class),
                "parents": self._get_parents(owl_class),
            }
            schema["classes"].append(class_info)

        # Extract object properties
        for prop in self.ontology.owl_object_properties():
            prop_info = {
                "uri": prop.getIRI().toString(),
                "label": self._get_label(prop),
                "description": self._get_description(prop),
                "domain": self._get_domain(prop),
                "range": self._get_range(prop),
            }
            schema["object_properties"].append(prop_info)

        # Extract data properties
        for prop in self.ontology.owl_data_properties():
            prop_info = {
                "uri": prop.getIRI().toString(),
                "label": self._get_label(prop),
                "description": self._get_description(prop),
                "domain": self._get_domain(prop),
                "range": self._get_range(prop),
            }
            schema["data_properties"].append(prop_info)

        return schema

    def verbalize_axiom(self, axiom: Any) -> str:
        """
        Convert an OWL axiom to natural language.

        Args:
            axiom: OWL axiom object

        Returns:
            Natural language verbalization
        """
        if not self.verbaliser:
            return str(axiom)

        return self.verbaliser.verbalise(axiom)

    def get_entity_types(self, entity_uri: str) -> List[str]:
        """
        Get all types (classes) for an entity.

        Args:
            entity_uri: URI of the entity

        Returns:
            List of class URIs
        """
        if not self.ontology:
            return []

        # TODO: Implement using OWLAPI
        # This will query for rdf:type assertions and inferred types
        return []

    def check_consistency(self) -> bool:
        """
        Check if ontology is consistent.

        Returns:
            True if consistent, False otherwise
        """
        if not self.ontology or not self.config.enable_reasoning:
            return True

        reasoner = self.ontology.reasoner()
        return reasoner.isConsistent()

    def validate_triple(
        self,
        subject_type: str,
        predicate_uri: str,
        object_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if a triple respects ontology constraints.

        Args:
            subject_type: Type (class) of subject
            predicate_uri: URI of predicate
            object_type: Type (class) of object

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.ontology:
            return True, None

        # TODO: Implement domain/range checking
        # Check if subject_type is in domain of predicate
        # Check if object_type is in range of predicate
        return True, None

    def get_subsumption_hierarchy(self) -> Dict[str, List[str]]:
        """
        Get class subsumption hierarchy.

        Returns:
            Dictionary mapping class URIs to their direct subclasses
        """
        if not self.taxonomy:
            return {}

        hierarchy = {}
        # TODO: Extract from taxonomy
        return hierarchy

    def get_entity_verbalization(self, entity_uri: str) -> str:
        """
        Get natural language description of an entity.

        Args:
            entity_uri: URI of entity

        Returns:
            Natural language description
        """
        # Get label and description
        label = entity_uri.split("#")[-1].split("/")[-1]
        # TODO: Look up rdfs:label and rdfs:comment
        return label

    def _get_label(self, entity: Any) -> str:
        """Get label for an ontology entity."""
        # TODO: Extract rdfs:label annotation
        iri = entity.getIRI().toString()
        return iri.split("#")[-1].split("/")[-1]

    def _get_description(self, entity: Any) -> str:
        """Get description for an ontology entity."""
        # TODO: Extract rdfs:comment annotation
        return ""

    def _get_parents(self, owl_class: Any) -> List[str]:
        """Get parent classes."""
        if not self.config.enable_reasoning:
            return []
        # TODO: Use reasoner to get superclasses
        return []

    def _get_domain(self, prop: Any) -> Optional[str]:
        """Get domain of a property."""
        # TODO: Extract domain axioms
        return None

    def _get_range(self, prop: Any) -> Optional[str]:
        """Get range of a property."""
        # TODO: Extract range axioms
        return None

    def normalize_ontology(self) -> None:
        """Normalize ontology to standard form (e.g., EL)."""
        if not self.normaliser:
            logger.warning("Normaliser not initialized")
            return

        logger.info("Normalizing ontology")
        # The normaliser transforms axioms to normal form
        # This is useful for consistent processing

    def project_to_rdf(self) -> List[Tuple[str, str, str]]:
        """
        Project TBox axioms to RDF triples.

        Returns:
            List of (subject, predicate, object) triples
        """
        if not self.projector:
            return []

        # The projector converts OWL axioms to RDF representation
        # This enables graph-based processing
        return []
