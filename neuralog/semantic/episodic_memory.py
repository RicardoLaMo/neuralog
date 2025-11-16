"""
Episodic Memory with Transformer architecture.

Inspired by episodic-transformer-memory-ppo for maintaining bounded,
attention-based memory across long sequences.

Instead of processing documents in isolated chunks, episodic memory:
1. Maintains sliding window of recent observations
2. Uses transformer self-attention for context
3. Enables efficient long-range dependencies
4. Tracks entity and event evolution
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class MemoryState:
    """
    Episodic memory state at a timestep.

    Stores:
    - Observation embedding
    - Hidden state from transformer
    - Attention scores
    - Metadata (timestamp, entities, etc.)
    """
    timestep: int
    observation: Any  # Original observation (text, event, etc.)
    embedding: np.ndarray  # Embedding vector
    hidden_state: Optional[np.ndarray] = None  # Transformer hidden state
    attention_weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for transformer.

    Enables the transformer to understand temporal relationships
    without absolute position encodings.
    """

    def __init__(self, dim: int, max_len: int = 512):
        """
        Initialize relative positional encoding.

        Args:
            dim: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        # Learnable relative position embeddings
        self.rel_pos_emb = nn.Parameter(
            torch.randn(2 * max_len - 1, dim)
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative position encodings.

        Args:
            seq_len: Sequence length

        Returns:
            Relative position embeddings [seq_len, seq_len, dim]
        """
        positions = torch.arange(seq_len, device=self.rel_pos_emb.device)
        relative_pos = positions[:, None] - positions[None, :]  # [seq_len, seq_len]

        # Shift to positive indices
        relative_pos = relative_pos + self.max_len - 1

        # Clamp to valid range
        relative_pos = torch.clamp(relative_pos, 0, 2 * self.max_len - 2)

        # Get embeddings
        rel_emb = self.rel_pos_emb[relative_pos]  # [seq_len, seq_len, dim]

        return rel_emb


class TransformerMemory(nn.Module):
    """
    Transformer-based episodic memory.

    Uses TransformerXL-style architecture with:
    - Multi-head self-attention
    - Relative positional encodings
    - Layer normalization
    - Sliding window memory
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        memory_window: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer Memory.

        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            memory_window: Size of sliding memory window
            dropout: Dropout rate
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TransformerMemory")

        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.memory_window = memory_window

        # Relative positional encoding
        self.pos_encoding = RelativePositionalEncoding(
            embedding_dim,
            max_len=memory_window
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        logger.info(
            f"TransformerMemory initialized: dim={embedding_dim}, "
            f"heads={num_heads}, layers={num_layers}, window={memory_window}"
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence with episodic memory.

        Args:
            embeddings: Input embeddings [batch, seq_len, dim]
            memory: Previous memory states [batch, mem_len, dim]

        Returns:
            Tuple of (output, updated_memory)
        """
        batch_size, seq_len, _ = embeddings.shape

        # Concatenate with memory if available
        if memory is not None:
            # Sliding window: keep only recent memory
            mem_len = min(memory.size(1), self.memory_window - seq_len)
            memory = memory[:, -mem_len:, :]
            inputs = torch.cat([memory, embeddings], dim=1)
        else:
            inputs = embeddings

        # Add relative positional encoding
        # (Simplified - actual implementation would integrate into attention)
        x = inputs

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Extract output and updated memory
        output = x[:, -seq_len:, :]  # Only return new outputs
        updated_memory = x  # Entire sequence becomes memory

        return output, updated_memory


class EpisodicMemory:
    """
    Episodic memory system for maintaining context across long documents.

    Key features:
    - Sliding window of recent observations
    - Transformer-based context integration
    - Efficient attention over relevant history
    - Entity and event tracking

    This replaces naive chunking with intelligent context management.
    """

    def __init__(
        self,
        embedding_model,
        embedding_dim: int = 256,
        memory_window: int = 128,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        """
        Initialize Episodic Memory.

        Args:
            embedding_model: Model for generating embeddings
            embedding_dim: Dimension of embeddings
            memory_window: Size of sliding window
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.memory_window = memory_window

        # Memory states
        self.memory_states: List[MemoryState] = []
        self.current_timestep = 0

        # Transformer memory (if PyTorch available)
        self.transformer_memory = None
        self.device = None

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transformer_memory = TransformerMemory(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                memory_window=memory_window
            ).to(self.device)
            self.transformer_memory.eval()

        logger.info(f"EpisodicMemory initialized (window={memory_window})")

    def add_observation(
        self,
        observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryState:
        """
        Add observation to episodic memory.

        Args:
            observation: Observation to add (text, event, etc.)
            metadata: Optional metadata

        Returns:
            Memory state
        """
        # Generate embedding
        if isinstance(observation, str):
            embedding = self.embedding_model.encode(observation)
        else:
            # Assume observation is already embedded
            embedding = observation

        # Create memory state
        state = MemoryState(
            timestep=self.current_timestep,
            observation=observation,
            embedding=embedding,
            metadata=metadata or {}
        )

        # Add to memory
        self.memory_states.append(state)

        # Maintain sliding window
        if len(self.memory_states) > self.memory_window:
            self.memory_states = self.memory_states[-self.memory_window:]

        self.current_timestep += 1

        # Update transformer hidden states if available
        if self.transformer_memory is not None:
            self._update_transformer_states()

        return state

    def _update_transformer_states(self):
        """Update transformer hidden states for all memory states."""

        if not self.memory_states:
            return

        # Gather embeddings
        embeddings = np.array([s.embedding for s in self.memory_states])
        embeddings_tensor = torch.tensor(
            embeddings,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension

        # Process through transformer
        with torch.no_grad():
            output, _ = self.transformer_memory(embeddings_tensor)

        # Update hidden states
        output_np = output.squeeze(0).cpu().numpy()
        for i, state in enumerate(self.memory_states):
            state.hidden_state = output_np[i]

    def get_context(
        self,
        query: Optional[str] = None,
        top_k: int = 10
    ) -> List[MemoryState]:
        """
        Retrieve relevant context from episodic memory.

        If query is provided, returns most relevant memory states.
        Otherwise, returns recent memory states.

        Args:
            query: Optional query for retrieval
            top_k: Number of states to return

        Returns:
            List of relevant memory states
        """
        if not self.memory_states:
            return []

        if query is None:
            # Return most recent states
            return self.memory_states[-top_k:]

        # Compute relevance using embeddings
        query_emb = self.embedding_model.encode(query)

        # Compute similarities
        similarities = []
        for state in self.memory_states:
            # Use hidden state if available (has context), else embedding
            state_emb = state.hidden_state if state.hidden_state is not None else state.embedding
            sim = np.dot(query_emb, state_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(state_emb)
            )
            similarities.append((state, sim))

        # Sort by relevance
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [s for s, _ in similarities[:top_k]]

    def get_entity_context(
        self,
        entity_id: str,
        temporal_window: Optional[int] = None
    ) -> List[MemoryState]:
        """
        Get all memory states involving a specific entity.

        Enables tracking entity evolution across time.

        Args:
            entity_id: Entity identifier
            temporal_window: Optional window size (recent N states)

        Returns:
            List of memory states involving entity
        """
        relevant_states = [
            s for s in self.memory_states
            if entity_id in s.metadata.get("entities", [])
        ]

        if temporal_window:
            relevant_states = relevant_states[-temporal_window:]

        return relevant_states

    def reset(self):
        """Clear episodic memory."""
        self.memory_states = []
        self.current_timestep = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "num_states": len(self.memory_states),
            "current_timestep": self.current_timestep,
            "memory_window": self.memory_window,
            "using_transformer": self.transformer_memory is not None,
        }
