"""Configuration management for NeuraLog."""

from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import yaml


class LLMConfig(BaseSettings):
    """Configuration for LLM integration."""
    provider: str = "openai"  # openai, anthropic, ollama
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.95

    class Config:
        env_prefix = "NEURALOG_LLM_"


class OntologyConfig(BaseSettings):
    """Configuration for ontology processing."""
    reasoner: str = "elk"  # elk, hermit, pellet
    owl_profile: str = "OWL2_EL"  # OWL2_EL, OWL2_QL, OWL2_RL, OWL2_DL
    enable_reasoning: bool = True
    max_axioms: int = 100000

    class Config:
        env_prefix = "NEURALOG_ONTOLOGY_"


class EmbeddingConfig(BaseSettings):
    """Configuration for embeddings."""
    model: str = "sentence-transformers/all-mpnet-base-v2"
    dimension: int = 768
    batch_size: int = 32
    device: str = "cpu"  # cpu, cuda, mps

    # Graph embedding specific
    walk_length: int = 80
    num_walks: int = 10
    window_size: int = 10
    embedding_dim: int = 128
    workers: int = 4

    class Config:
        env_prefix = "NEURALOG_EMBEDDING_"


class VerificationConfig(BaseSettings):
    """Configuration for formal verification."""
    enable_verification: bool = True
    solver: str = "z3"  # z3, cvc5
    timeout: int = 30  # seconds
    soundness_threshold: float = 0.99

    class Config:
        env_prefix = "NEURALOG_VERIFICATION_"


class StorageConfig(BaseSettings):
    """Configuration for data storage."""
    # Triple store
    triple_store: str = "memory"  # memory, jena, rdfox
    sparql_endpoint: Optional[str] = None

    # Vector store
    vector_store: str = "chroma"  # chroma, qdrant, faiss
    vector_store_path: Path = Path("./data/vector_store")

    # Document store
    document_store_path: Path = Path("./data/documents")

    class Config:
        env_prefix = "NEURALOG_STORAGE_"


class ExtractionConfig(BaseSettings):
    """Configuration for information extraction."""
    confidence_threshold: float = 0.7
    max_entities_per_document: int = 1000
    max_triples_per_document: int = 5000
    enable_coreference: bool = True
    enable_entity_linking: bool = True

    class Config:
        env_prefix = "NEURALOG_EXTRACTION_"


class Config(BaseSettings):
    """Main configuration for NeuraLog."""
    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    # General settings
    debug: bool = False
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # Paths
    base_dir: Path = Path(".")
    models_dir: Path = Path("./models")
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./.cache")

    class Config:
        env_prefix = "NEURALOG_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def setup_directories(self) -> None:
        """Create necessary directories."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.storage.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.storage.document_store_path.mkdir(parents=True, exist_ok=True)
