"""
NeuraLog extensions module for custom implementations.

This module provides base classes and interfaces for extending NeuraLog
with custom LLM providers, ontology loaders, extractors, and more.

Classes:
    LLMProvider: Base class for custom LLM implementations
    OntologyProvider: Base class for custom ontology loaders
    ExtractorPlugin: Base class for custom extractors
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class LLMProvider(ABC):
    """Base class for LLM provider implementations.

    All custom LLM providers should inherit from this class and implement
    the required abstract methods.

    Example:
        >>> class MyLLMProvider(LLMProvider):
        ...     def generate(self, prompt: str, **kwargs) -> str:
        ...         return "generated text"
        ...
        ...     def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        ...         return [self.generate(p, **kwargs) for p in prompts]
        ...
        ...     def count_tokens(self, text: str) -> int:
        ...         return len(text.split())
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt for generation
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            Generated text response

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Additional arguments

        Returns:
            List of generated text responses

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Useful for cost estimation and rate limiting.

        Args:
            text: Input text to count tokens for

        Returns:
            Number of tokens

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        pass


class OntologyProvider(ABC):
    """Base class for ontology provider implementations.

    All custom ontology loaders should inherit from this class.

    Example:
        >>> class MyOntologyProvider(OntologyProvider):
        ...     def load(self, source: str):
        ...         # Custom loading logic
        ...         pass
    """

    @abstractmethod
    def load(self, source: str) -> Any:
        """Load ontology from source.

        Args:
            source: Source identifier (path, URL, database reference, etc.)

        Returns:
            Loaded ontology graph object

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        pass


class ExtractorPlugin(ABC):
    """Base class for custom extractor implementations.

    All custom extractors should inherit from this class and implement
    entity and relation extraction methods.

    Example:
        >>> from neuralog.core.types import Entity, Relation
        >>> class MyExtractor(ExtractorPlugin):
        ...     def extract_entities(self, text: str) -> List[Entity]:
        ...         # Custom entity extraction
        ...         return []
        ...
        ...     def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        ...         # Custom relation extraction
        ...         return []
    """

    @abstractmethod
    def extract_entities(self, text: str) -> List[Any]:
        """Extract entities from text.

        Args:
            text: Input text to extract entities from

        Returns:
            List of Entity objects

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        pass

    @abstractmethod
    def extract_relations(self, text: str, entities: List[Any]) -> List[Any]:
        """Extract relations between entities.

        Args:
            text: Input text to extract relations from
            entities: List of Entity objects identified in the text

        Returns:
            List of Relation objects

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        pass


__all__ = ["LLMProvider", "OntologyProvider", "ExtractorPlugin"]
