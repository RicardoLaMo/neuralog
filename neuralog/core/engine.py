"""Core engine orchestrating neural-symbolic operations."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from loguru import logger

from neuralog.core.config import Config
from neuralog.core.types import (
    ExtractionResult,
    KnowledgeGraph,
    Triple,
    Entity,
    ConfidenceLevel,
)


class Engine:
    """
    Main NeuraLog engine coordinating all components.

    The engine orchestrates the neural-symbolic pipeline:
    1. Text input processing
    2. Neural extraction (LLM-based)
    3. Symbolic validation (ontology-based)
    4. Hybrid reasoning
    5. Optional formal verification
    6. Knowledge graph construction and storage
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the NeuraLog engine.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or Config()
        self.config.setup_directories()

        # Initialize components (lazy loading)
        self._ontology_manager = None
        self._graph_embedder = None
        self._llm_interface = None
        self._embedding_model = None
        self._neurosymbolic_reasoner = None
        self._kg_distiller = None
        self._verifier = None
        self._triple_store = None
        self._vector_store = None

        logger.info(f"NeuraLog Engine initialized with config: {self.config}")

    @property
    def ontology_manager(self):
        """Lazy load ontology manager."""
        if self._ontology_manager is None:
            from neuralog.symbolic import OntologyManager
            self._ontology_manager = OntologyManager(self.config.ontology)
        return self._ontology_manager

    @property
    def graph_embedder(self):
        """Lazy load graph embedder."""
        if self._graph_embedder is None:
            from neuralog.symbolic import GraphEmbedder
            self._graph_embedder = GraphEmbedder(self.config.embedding)
        return self._graph_embedder

    @property
    def llm_interface(self):
        """Lazy load LLM interface."""
        if self._llm_interface is None:
            from neuralog.neural import LLMInterface
            self._llm_interface = LLMInterface(self.config.llm)
        return self._llm_interface

    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from neuralog.neural import EmbeddingModel
            self._embedding_model = EmbeddingModel(self.config.embedding)
        return self._embedding_model

    @property
    def neurosymbolic_reasoner(self):
        """Lazy load neurosymbolic reasoner."""
        if self._neurosymbolic_reasoner is None:
            from neuralog.integration import NeurosymbolicReasoner
            self._neurosymbolic_reasoner = NeurosymbolicReasoner(
                self.ontology_manager,
                self.llm_interface,
                self.config
            )
        return self._neurosymbolic_reasoner

    @property
    def kg_distiller(self):
        """Lazy load KG distiller."""
        if self._kg_distiller is None:
            from neuralog.integration import KGDistiller
            self._kg_distiller = KGDistiller(
                self.llm_interface,
                self.ontology_manager,
                self.embedding_model,
                self.config
            )
        return self._kg_distiller

    @property
    def verifier(self):
        """Lazy load formal verifier."""
        if self._verifier is None:
            from neuralog.integration import FormalVerifier
            self._verifier = FormalVerifier(self.config.verification)
        return self._verifier

    def load_ontology(self, ontology_path: Union[str, Path]) -> None:
        """
        Load an ontology for domain knowledge.

        Args:
            ontology_path: Path to OWL/RDF ontology file
        """
        logger.info(f"Loading ontology from {ontology_path}")
        self.ontology_manager.load_ontology(ontology_path)

    def extract_knowledge(
        self,
        text: str,
        verify: bool = False,
        confidence_threshold: Optional[float] = None
    ) -> ExtractionResult:
        """
        Extract knowledge from text using neural-symbolic pipeline.

        Pipeline:
        1. LLM-based entity and relation extraction
        2. Ontology-guided validation and typing
        3. Neurosymbolic reasoning for consistency
        4. Optional formal verification for critical facts
        5. Confidence scoring and filtering

        Args:
            text: Input text for extraction
            verify: Whether to apply formal verification
            confidence_threshold: Minimum confidence (uses config default if None)

        Returns:
            ExtractionResult with extracted triples, entities, and metadata
        """
        logger.info(f"Extracting knowledge from text (length: {len(text)})")

        threshold = confidence_threshold or self.config.extraction.confidence_threshold

        # Stage 1: Neural extraction via KG Distiller
        logger.debug("Stage 1: Neural extraction")
        extraction_result = self.kg_distiller.extract_from_text(text)

        # Stage 2: Neurosymbolic reasoning
        logger.debug("Stage 2: Neurosymbolic reasoning")
        validated_result = self.neurosymbolic_reasoner.validate_extraction(
            extraction_result
        )

        # Stage 3: Optional formal verification
        if verify:
            logger.debug("Stage 3: Formal verification")
            verified_result = self.verifier.verify_extraction(validated_result)
            return verified_result

        # Filter by confidence threshold
        filtered_triples = [
            t for t in validated_result.triples
            if t.confidence >= threshold
        ]
        validated_result.triples = filtered_triples

        return validated_result

    def extract_to_kg(
        self,
        text: str,
        kg_name: Optional[str] = None,
        **kwargs
    ) -> KnowledgeGraph:
        """
        Extract knowledge and build a knowledge graph.

        Args:
            text: Input text
            kg_name: Name for the knowledge graph
            **kwargs: Additional arguments for extract_knowledge

        Returns:
            KnowledgeGraph object
        """
        result = self.extract_knowledge(text, **kwargs)
        kg_name = kg_name or f"kg_{hash(text)}"
        return result.to_knowledge_graph(kg_name)

    def query(
        self,
        query: str,
        kg: Optional[KnowledgeGraph] = None,
        use_reasoning: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph using natural language or SPARQL.

        Args:
            query: Natural language or SPARQL query
            kg: Knowledge graph to query (uses stored KG if None)
            use_reasoning: Whether to apply reasoning during query

        Returns:
            List of query results
        """
        logger.info(f"Processing query: {query}")

        # Determine if query is SPARQL or natural language
        is_sparql = query.strip().upper().startswith("SELECT")

        if is_sparql:
            # Direct SPARQL query
            return self._execute_sparql(query, kg)
        else:
            # Natural language query - convert to SPARQL via LLM
            sparql_query = self.llm_interface.nl_to_sparql(
                query,
                ontology=self.ontology_manager.get_schema()
            )
            results = self._execute_sparql(sparql_query, kg)

            if use_reasoning:
                # Enhance results with neurosymbolic reasoning
                results = self.neurosymbolic_reasoner.enhance_results(results)

            return results

    def _execute_sparql(
        self,
        sparql: str,
        kg: Optional[KnowledgeGraph] = None
    ) -> List[Dict[str, Any]]:
        """Execute SPARQL query against knowledge graph."""
        # TODO: Implement SPARQL execution
        # This will use the triple store when implemented
        raise NotImplementedError("SPARQL execution not yet implemented")

    def build_embeddings(
        self,
        kg: KnowledgeGraph,
        method: str = "rdf_walk"
    ) -> Dict[str, List[float]]:
        """
        Build embeddings for knowledge graph entities.

        Args:
            kg: Knowledge graph
            method: Embedding method (rdf_walk, node2vec, transe, etc.)

        Returns:
            Dictionary mapping entity URIs to embedding vectors
        """
        logger.info(f"Building embeddings for KG '{kg.name}' using method: {method}")
        return self.graph_embedder.embed_graph(kg, method=method)

    def save_kg(
        self,
        kg: KnowledgeGraph,
        path: Union[str, Path],
        format: str = "turtle"
    ) -> None:
        """
        Save knowledge graph to file.

        Args:
            kg: Knowledge graph to save
            path: Output file path
            format: RDF serialization format (turtle, xml, nt, etc.)
        """
        logger.info(f"Saving KG '{kg.name}' to {path} in {format} format")
        # TODO: Implement KG serialization
        raise NotImplementedError("KG saving not yet implemented")

    def load_kg(
        self,
        path: Union[str, Path],
        format: str = "turtle"
    ) -> KnowledgeGraph:
        """
        Load knowledge graph from file.

        Args:
            path: Input file path
            format: RDF serialization format

        Returns:
            Loaded KnowledgeGraph
        """
        logger.info(f"Loading KG from {path}")
        # TODO: Implement KG loading
        raise NotImplementedError("KG loading not yet implemented")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current engine state.

        Returns:
            Dictionary with various statistics
        """
        stats = {
            "config": {
                "llm_model": self.config.llm.model,
                "embedding_model": self.config.embedding.model,
                "reasoner": self.config.ontology.reasoner,
                "verification_enabled": self.config.verification.enable_verification,
            },
            "components": {
                "ontology_loaded": self._ontology_manager is not None,
                "llm_initialized": self._llm_interface is not None,
                "embedder_initialized": self._graph_embedder is not None,
            }
        }

        if self._ontology_manager:
            stats["ontology"] = self.ontology_manager.get_statistics()

        return stats
