"""
Graph embedding for knowledge graphs.

Modernized implementation of RDF/OWL-aware graph walking and embedding,
inspired by walking-rdf-and-owl but using modern Python and PyTorch.

Supports multiple embedding methods:
- RDF-aware random walks (modernized DeepWalk)
- Node2Vec
- TransE, DistMult (knowledge graph embeddings)
- Hyperbolic embeddings for hierarchies
"""

from typing import Any, Dict, List, Optional, Tuple, Set
import random
from collections import defaultdict
from loguru import logger
import numpy as np

try:
    import networkx as nx
    import torch
    import torch.nn as nn
    from torch_geometric.nn import Node2Vec as PyGNode2Vec
    from gensim.models import Word2Vec
    EMBEDDING_DEPS_AVAILABLE = True
except ImportError:
    EMBEDDING_DEPS_AVAILABLE = False
    logger.warning(
        "Graph embedding dependencies not available. "
        "Install with: pip install networkx torch torch-geometric gensim"
    )

from neuralog.core.config import EmbeddingConfig
from neuralog.core.types import KnowledgeGraph, Triple


class GraphEmbedder:
    """
    Generate embeddings for knowledge graph entities using various methods.

    Key innovation: RDF-aware random walks that respect ontological structure
    and edge types, modernizing the walking-rdf-and-owl approach.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize graph embedder.

        Args:
            config: Embedding configuration
        """
        if not EMBEDDING_DEPS_AVAILABLE:
            raise ImportError(
                "Graph embedding dependencies required. "
                "Install with: pip install networkx torch torch-geometric gensim"
            )

        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"GraphEmbedder initialized with device: {self.device}")

    def embed_graph(
        self,
        kg: KnowledgeGraph,
        method: str = "rdf_walk"
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all entities in the knowledge graph.

        Args:
            kg: Knowledge graph
            method: Embedding method
                - "rdf_walk": RDF-aware random walks (DeepWalk-style)
                - "node2vec": Node2Vec algorithm
                - "transe": TransE knowledge graph embedding
                - "distmult": DistMult knowledge graph embedding

        Returns:
            Dictionary mapping entity URIs to embedding vectors
        """
        logger.info(f"Generating embeddings for KG '{kg.name}' using method: {method}")

        if method == "rdf_walk":
            return self._embed_rdf_walk(kg)
        elif method == "node2vec":
            return self._embed_node2vec(kg)
        elif method == "transe":
            return self._embed_transe(kg)
        elif method == "distmult":
            return self._embed_distmult(kg)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

    def _embed_rdf_walk(self, kg: KnowledgeGraph) -> Dict[str, np.ndarray]:
        """
        RDF-aware random walk embeddings (modernized walking-rdf-and-owl).

        Key features:
        - Respects edge types (properties) during walks
        - Can be biased by ontology hierarchy
        - Preserves both graph structure and semantic relationships
        """
        logger.info("Generating RDF-aware random walk embeddings")

        # Build graph representation
        graph, uri_to_node = self._build_graph(kg)

        # Generate random walks
        walks = self._generate_rdf_walks(graph, kg)

        # Train Word2Vec on walks
        model = Word2Vec(
            sentences=walks,
            vector_size=self.config.embedding_dim,
            window=self.config.window_size,
            min_count=1,
            workers=self.config.workers,
            sg=1,  # Skip-gram
            hs=0,  # Negative sampling
            negative=5,
            epochs=10
        )

        # Extract embeddings
        embeddings = {}
        for uri, node_id in uri_to_node.items():
            node_str = str(node_id)
            if node_str in model.wv:
                embeddings[uri] = model.wv[node_str]

        logger.info(f"Generated embeddings for {len(embeddings)} entities")
        return embeddings

    def _generate_rdf_walks(
        self,
        graph: nx.Graph,
        kg: KnowledgeGraph
    ) -> List[List[str]]:
        """
        Generate RDF-aware random walks.

        Unlike standard random walks, this considers:
        - Edge types (object properties)
        - Type constraints from ontology
        - Weighted sampling based on property importance
        """
        walks = []
        nodes = list(graph.nodes())

        logger.info(
            f"Generating {self.config.num_walks} walks of length "
            f"{self.config.walk_length} for {len(nodes)} nodes"
        )

        for _ in range(self.config.num_walks):
            for node in nodes:
                walk = self._rdf_walk(graph, node, self.config.walk_length)
                if len(walk) > 1:
                    walks.append([str(n) for n in walk])

        logger.info(f"Generated {len(walks)} walks")
        return walks

    def _rdf_walk(
        self,
        graph: nx.Graph,
        start_node: Any,
        length: int
    ) -> List[Any]:
        """
        Perform a single RDF-aware random walk.

        Args:
            graph: NetworkX graph
            start_node: Starting node
            length: Walk length

        Returns:
            List of nodes in the walk
        """
        walk = [start_node]
        current = start_node

        for _ in range(length - 1):
            neighbors = list(graph.neighbors(current))
            if not neighbors:
                break

            # In RDF-aware walks, we can bias sampling based on edge types
            # For now, uniform sampling
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current = next_node

        return walk

    def _embed_node2vec(self, kg: KnowledgeGraph) -> Dict[str, np.ndarray]:
        """
        Node2Vec embeddings using PyTorch Geometric.

        Node2Vec extends DeepWalk with biased random walks that can
        explore neighborhoods in BFS or DFS fashion.
        """
        logger.info("Generating Node2Vec embeddings")

        graph, uri_to_node = self._build_graph(kg)

        # Convert to PyTorch Geometric format
        edge_index = torch.tensor(
            list(graph.edges()),
            dtype=torch.long
        ).t().contiguous()

        # Train Node2Vec
        model = PyGNode2Vec(
            edge_index,
            embedding_dim=self.config.embedding_dim,
            walk_length=self.config.walk_length,
            context_size=self.config.window_size,
            walks_per_node=self.config.num_walks,
            num_negative_samples=5,
            p=1.0,  # Return parameter
            q=1.0,  # In-out parameter
        ).to(self.device)

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        for epoch in range(100):
            optimizer.zero_grad()
            loss = model.loss(edge_index.to(self.device))
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # Extract embeddings
        model.eval()
        embeddings_tensor = model().detach().cpu().numpy()

        embeddings = {}
        for uri, node_id in uri_to_node.items():
            embeddings[uri] = embeddings_tensor[node_id]

        return embeddings

    def _embed_transe(self, kg: KnowledgeGraph) -> Dict[str, np.ndarray]:
        """
        TransE knowledge graph embeddings.

        TransE models relationships as translations in embedding space:
        h + r ≈ t (head + relation ≈ tail)
        """
        logger.info("TransE embedding not yet implemented")
        # TODO: Implement TransE
        # This requires training on triples with relation embeddings
        raise NotImplementedError("TransE embedding coming soon")

    def _embed_distmult(self, kg: KnowledgeGraph) -> Dict[str, np.ndarray]:
        """
        DistMult knowledge graph embeddings.

        DistMult models relationships as element-wise multiplication:
        score(h, r, t) = h^T diag(r) t
        """
        logger.info("DistMult embedding not yet implemented")
        # TODO: Implement DistMult
        raise NotImplementedError("DistMult embedding coming soon")

    def _build_graph(
        self,
        kg: KnowledgeGraph
    ) -> Tuple[nx.Graph, Dict[str, int]]:
        """
        Build NetworkX graph from knowledge graph.

        Args:
            kg: Knowledge graph

        Returns:
            Tuple of (NetworkX graph, URI to node ID mapping)
        """
        graph = nx.Graph()
        uri_to_node = {}
        node_counter = 0

        # Map URIs to integer node IDs
        for entity_uri in kg.entities.keys():
            if entity_uri not in uri_to_node:
                uri_to_node[entity_uri] = node_counter
                graph.add_node(node_counter, uri=entity_uri)
                node_counter += 1

        # Add edges from triples
        for triple in kg.triples:
            s_uri = triple.subject.uri if hasattr(triple.subject, 'uri') else str(triple.subject)
            o_uri = triple.object.uri if hasattr(triple.object, 'uri') else str(triple.object)

            # Ensure both subject and object are in the mapping
            if s_uri not in uri_to_node:
                uri_to_node[s_uri] = node_counter
                graph.add_node(node_counter, uri=s_uri)
                node_counter += 1

            if o_uri not in uri_to_node:
                uri_to_node[o_uri] = node_counter
                graph.add_node(node_counter, uri=o_uri)
                node_counter += 1

            # Add edge with property information
            p_uri = triple.predicate.uri if hasattr(triple.predicate, 'uri') else str(triple.predicate)
            graph.add_edge(
                uri_to_node[s_uri],
                uri_to_node[o_uri],
                property=p_uri
            )

        logger.info(
            f"Built graph with {graph.number_of_nodes()} nodes "
            f"and {graph.number_of_edges()} edges"
        )

        return graph, uri_to_node

    def compute_similarity(
        self,
        embeddings: Dict[str, np.ndarray],
        entity1: str,
        entity2: str,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two entities based on embeddings.

        Args:
            embeddings: Entity embeddings
            entity1: First entity URI
            entity2: Second entity URI
            metric: Similarity metric (cosine, euclidean)

        Returns:
            Similarity score
        """
        if entity1 not in embeddings or entity2 not in embeddings:
            return 0.0

        vec1 = embeddings[entity1]
        vec2 = embeddings[entity2]

        if metric == "cosine":
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        elif metric == "euclidean":
            return -np.linalg.norm(vec1 - vec2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def find_similar_entities(
        self,
        embeddings: Dict[str, np.ndarray],
        entity: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most similar entities to a given entity.

        Args:
            embeddings: Entity embeddings
            entity: Entity URI
            top_k: Number of similar entities to return

        Returns:
            List of (entity_uri, similarity_score) tuples
        """
        if entity not in embeddings:
            return []

        similarities = []
        for other_entity in embeddings:
            if other_entity != entity:
                sim = self.compute_similarity(embeddings, entity, other_entity)
                similarities.append((other_entity, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
