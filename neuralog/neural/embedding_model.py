"""Text embedding models for semantic similarity and retrieval."""

from typing import List, Union
from loguru import logger
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from neuralog.core.config import EmbeddingConfig


class EmbeddingModel:
    """
    Text embedding model for semantic similarity.

    Uses sentence-transformers for efficient embeddings with
    support for various pre-trained models.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding model.

        Args:
            config: Embedding configuration
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )

        self.config = config
        self.model = SentenceTransformer(
            config.model,
            device=config.device
        )

        logger.info(f"EmbeddingModel loaded: {config.model}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding (uses config default if None)
            show_progress: Show progress bar

        Returns:
            Embedding vector(s)
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score
        """
        emb1, emb2 = self.encode([text1, text2])
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts to query.

        Args:
            query: Query text
            candidates: Candidate texts
            top_k: Number of results to return

        Returns:
            List of (text, score) tuples
        """
        query_emb = self.encode(query)
        candidate_embs = self.encode(candidates)

        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb) / (
            np.linalg.norm(candidate_embs, axis=1) * np.linalg.norm(query_emb)
        )

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(candidates[i], similarities[i]) for i in top_indices]

        return results
