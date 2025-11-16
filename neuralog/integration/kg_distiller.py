"""
Knowledge Graph Distiller - Reliable KG construction from unstructured text.

Inspired by GraphMERT approach (arxiv:2510.09580):
- Efficient and scalable distillation
- Ontology-consistent generation
- High factual accuracy (FActScore)
- LLM-guided with symbolic validation
"""

from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import re

from neuralog.core.types import (
    ExtractionResult,
    Triple,
    Entity,
    Relation,
    EntityType,
    ConfidenceLevel
)
from neuralog.core.config import Config


class KGDistiller:
    """
    Distills reliable, ontology-consistent knowledge graphs from text.

    Multi-stage pipeline:
    1. Text preprocessing and chunking
    2. LLM-based entity extraction with ontology guidance
    3. LLM-based relation extraction with type constraints
    4. Triple generation and candidate scoring
    5. Ontological validation and filtering
    6. Iterative refinement with feedback
    7. Quality assessment (FActScore, ontology compliance)
    """

    def __init__(
        self,
        llm_interface,
        ontology_manager,
        embedding_model,
        config: Config
    ):
        """
        Initialize KG distiller.

        Args:
            llm_interface: LLM interface for extraction
            ontology_manager: Ontology manager for validation
            embedding_model: Embedding model for similarity
            config: Configuration
        """
        self.llm_interface = llm_interface
        self.ontology_manager = ontology_manager
        self.embedding_model = embedding_model
        self.config = config

        logger.info("KGDistiller initialized")

    def extract_from_text(
        self,
        text: str,
        max_entities: Optional[int] = None,
        max_triples: Optional[int] = None
    ) -> ExtractionResult:
        """
        Extract knowledge graph from unstructured text.

        Args:
            text: Input text
            max_entities: Maximum entities to extract
            max_triples: Maximum triples to extract

        Returns:
            ExtractionResult with entities, relations, and triples
        """
        logger.info(f"Extracting KG from text (length: {len(text)})")

        max_entities = max_entities or self.config.extraction.max_entities_per_document
        max_triples = max_triples or self.config.extraction.max_triples_per_document

        # Stage 1: Preprocessing
        chunks = self._preprocess_text(text)

        # Stage 2: Entity extraction
        entities = self._extract_entities(chunks)
        entities = entities[:max_entities]

        # Stage 3: Relation extraction
        relations, triples = self._extract_relations(chunks, entities)

        triples = triples[:max_triples]

        # Stage 4: Validation and scoring
        triples = self._score_triples(triples, text)

        # Stage 5: Construct result
        provenance = {
            "source_text_length": len(text),
            "num_chunks": len(chunks),
            "extraction_method": "kg_distiller",
            "ontology": self.ontology_manager.ontology.iri if self.ontology_manager.ontology else None
        }

        # Compute overall confidence
        if triples:
            avg_confidence = sum(t.confidence for t in triples) / len(triples)
        else:
            avg_confidence = 0.0

        result = ExtractionResult(
            triples=triples,
            entities=list(entities),
            relations=list(set(relations)),
            confidence=avg_confidence,
            confidence_level=self._determine_confidence_level(avg_confidence),
            provenance=provenance
        )

        logger.info(
            f"Extraction complete: {len(entities)} entities, "
            f"{len(relations)} relation types, {len(triples)} triples, "
            f"confidence: {avg_confidence:.2f}"
        )

        return result

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text into chunks for extraction.

        Strategies:
        - Sentence splitting
        - Paragraph chunking
        - Sliding window with overlap

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        # Simple sentence splitting for now
        # TODO: Use more sophisticated chunking (e.g., semantic chunking)

        # Split by common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        logger.debug(f"Preprocessed text into {len(sentences)} chunks")
        return sentences

    def _extract_entities(self, chunks: List[str]) -> List[Entity]:
        """
        Extract entities from text chunks using ontology-guided LLM.

        Args:
            chunks: Text chunks

        Returns:
            List of extracted entities
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks")

        # Get ontology schema for guidance
        ontology_schema = self.ontology_manager.get_schema()

        entities = []
        entity_mentions = {}  # Track mentions for coreference

        for chunk in chunks:
            # Extract entities from chunk
            result = self.llm_interface.extract_entities_relations(
                text=chunk,
                ontology_schema=ontology_schema
            )

            # Process extracted entities
            for entity_data in result.get("entities", []):
                entity_text = entity_data.get("text", "")
                entity_type_label = entity_data.get("type", "")

                # Create URI (simple approach - hash of canonical form)
                entity_uri = self._create_entity_uri(entity_text)

                # Map type label to ontology class
                entity_type_uri = self._map_type_to_ontology(
                    entity_type_label,
                    ontology_schema
                )

                # Create or update entity
                if entity_uri in entity_mentions:
                    # Entity already seen - update confidence
                    entity_mentions[entity_uri]["count"] += 1
                else:
                    entity = Entity(
                        uri=entity_uri,
                        label=entity_text,
                        entity_type=EntityType.INDIVIDUAL,
                        attributes={
                            "ontology_type": entity_type_uri,
                            "mentions": [entity_text]
                        },
                        confidence=0.8,  # Initial confidence
                        source=chunk[:100]  # First 100 chars as source
                    )
                    entities.append(entity)
                    entity_mentions[entity_uri] = {"entity": entity, "count": 1}

        # Boost confidence for frequently mentioned entities
        for uri, data in entity_mentions.items():
            count = data["count"]
            entity = data["entity"]
            # Logarithmic boost for frequency
            frequency_boost = min(0.2, 0.05 * count)
            entity.confidence = min(1.0, entity.confidence + frequency_boost)

        logger.info(f"Extracted {len(entities)} unique entities")
        return entities

    def _extract_relations(
        self,
        chunks: List[str],
        entities: List[Entity]
    ) -> Tuple[List[Relation], List[Triple]]:
        """
        Extract relations between entities.

        Args:
            chunks: Text chunks
            entities: Extracted entities

        Returns:
            Tuple of (relations, triples)
        """
        logger.info(f"Extracting relations for {len(entities)} entities")

        ontology_schema = self.ontology_manager.get_schema()

        relations = []
        triples = []
        relation_types_seen = set()

        # Build entity lookup by label
        entity_by_label = {}
        for entity in entities:
            if entity.label:
                entity_by_label[entity.label.lower()] = entity

        for chunk in chunks:
            # Extract relations from chunk
            result = self.llm_interface.extract_entities_relations(
                text=chunk,
                ontology_schema=ontology_schema
            )

            # Process extracted relations
            for rel_data in result.get("relations", []):
                subject_text = rel_data.get("subject", "")
                predicate_label = rel_data.get("predicate", "")
                object_text = rel_data.get("object", "")

                # Find matching entities
                subject_entity = entity_by_label.get(subject_text.lower())
                object_entity = entity_by_label.get(object_text.lower())

                if not subject_entity or not object_entity:
                    logger.debug(f"Skipping relation - entities not found: {rel_data}")
                    continue

                # Map predicate to ontology property
                predicate_uri = self._map_relation_to_ontology(
                    predicate_label,
                    ontology_schema
                )

                # Create or find relation
                if predicate_uri not in relation_types_seen:
                    relation = Relation(
                        uri=predicate_uri,
                        label=predicate_label
                    )
                    relations.append(relation)
                    relation_types_seen.add(predicate_uri)

                # Create triple
                triple = Triple(
                    subject=subject_entity,
                    predicate=predicate_uri,
                    object=object_entity,
                    confidence=0.75,  # Initial confidence
                    provenance={
                        "source_chunk": chunk,
                        "extraction_method": "llm"
                    }
                )

                triples.append(triple)

        logger.info(f"Extracted {len(triples)} triples with {len(relations)} relation types")
        return relations, triples

    def _score_triples(self, triples: List[Triple], source_text: str) -> List[Triple]:
        """
        Score triples for reliability and factuality.

        Factors:
        - Entity confidence
        - Textual evidence strength
        - Ontological consistency
        - Multiple mentions

        Args:
            triples: Triples to score
            source_text: Original source text

        Returns:
            Scored triples
        """
        for triple in triples:
            # Base confidence from extraction
            base_confidence = triple.confidence

            # Factor 1: Entity confidence
            subject_conf = triple.subject.confidence if isinstance(triple.subject, Entity) else 1.0
            object_conf = triple.object.confidence if isinstance(triple.object, Entity) else 1.0
            entity_factor = (subject_conf + object_conf) / 2

            # Factor 2: Textual evidence (check if mentioned in source)
            evidence_factor = self._compute_evidence_score(triple, source_text)

            # Combine factors
            final_confidence = base_confidence * 0.5 + entity_factor * 0.3 + evidence_factor * 0.2
            triple.confidence = min(1.0, final_confidence)

            # Set confidence level
            triple.confidence_level = self._determine_confidence_level(triple.confidence)

        return triples

    def _compute_evidence_score(self, triple: Triple, source_text: str) -> float:
        """Compute evidence score based on source text."""
        # Simple approach: check if subject and object appear close together
        subject_label = triple.subject.label if isinstance(triple.subject, Entity) else str(triple.subject)
        object_label = triple.object.label if isinstance(triple.object, Entity) else str(triple.object)

        if not subject_label or not object_label:
            return 0.5

        # Find positions
        source_lower = source_text.lower()
        subject_lower = subject_label.lower()
        object_lower = object_label.lower()

        if subject_lower in source_lower and object_lower in source_lower:
            subject_pos = source_lower.index(subject_lower)
            object_pos = source_lower.index(object_lower)
            distance = abs(subject_pos - object_pos)

            # Closer mentions = higher evidence
            if distance < 50:
                return 1.0
            elif distance < 200:
                return 0.8
            else:
                return 0.6
        else:
            return 0.3

    def _create_entity_uri(self, entity_text: str) -> str:
        """Create URI for entity."""
        # Simple approach: namespace + normalized text
        normalized = entity_text.lower().replace(" ", "_")
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return f"http://neuralog.ai/entity/{normalized}"

    def _map_type_to_ontology(
        self,
        type_label: str,
        ontology_schema: Dict[str, Any]
    ) -> Optional[str]:
        """Map extracted type label to ontology class URI."""
        if not ontology_schema or "classes" not in ontology_schema:
            return None

        # Simple fuzzy matching
        type_label_lower = type_label.lower()
        for cls in ontology_schema["classes"]:
            cls_label = cls.get("label", "").lower()
            if cls_label == type_label_lower or type_label_lower in cls_label:
                return cls["uri"]

        return None

    def _map_relation_to_ontology(
        self,
        relation_label: str,
        ontology_schema: Dict[str, Any]
    ) -> str:
        """Map extracted relation label to ontology property URI."""
        if not ontology_schema or "object_properties" not in ontology_schema:
            # Fallback: create URI from label
            normalized = relation_label.lower().replace(" ", "_")
            return f"http://neuralog.ai/property/{normalized}"

        # Simple fuzzy matching
        relation_label_lower = relation_label.lower()
        for prop in ontology_schema["object_properties"]:
            prop_label = prop.get("label", "").lower()
            if prop_label == relation_label_lower or relation_label_lower in prop_label:
                return prop["uri"]

        # Fallback
        normalized = relation_label.lower().replace(" ", "_")
        return f"http://neuralog.ai/property/{normalized}"

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map confidence score to confidence level."""
        if confidence >= 0.99:
            return ConfidenceLevel.VERIFIED
        elif confidence >= 0.90:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def compute_fact_score(
        self,
        kg: "KnowledgeGraph",
        source_text: str
    ) -> float:
        """
        Compute FActScore for extracted knowledge graph.

        FActScore measures factual accuracy by checking if extracted
        facts are supported by the source text.

        Args:
            kg: Extracted knowledge graph
            source_text: Original source text

        Returns:
            FActScore (0-1, higher is better)
        """
        if not kg.triples:
            return 0.0

        supported_count = 0

        for triple in kg.triples:
            # Check if triple is supported by source text
            evidence_score = self._compute_evidence_score(triple, source_text)
            if evidence_score >= 0.7:
                supported_count += 1

        fact_score = supported_count / len(kg.triples)
        logger.info(f"FActScore: {fact_score:.2%} ({supported_count}/{len(kg.triples)} triples)")

        return fact_score
