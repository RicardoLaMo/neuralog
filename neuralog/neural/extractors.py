"""Entity and relation extractors."""

from typing import List, Dict, Any
from loguru import logger

from neuralog.core.types import Entity, Relation, Triple


class EntityExtractor:
    """Extract entities from text."""

    def __init__(self, llm_interface):
        self.llm_interface = llm_interface

    def extract(self, text: str, ontology_schema: Dict[str, Any]) -> List[Entity]:
        """Extract entities from text."""
        # This will use the LLM interface
        logger.info(f"Extracting entities from text (length: {len(text)})")
        # TODO: Implement entity extraction logic
        return []


class RelationExtractor:
    """Extract relations from text."""

    def __init__(self, llm_interface):
        self.llm_interface = llm_interface

    def extract(
        self,
        text: str,
        entities: List[Entity],
        ontology_schema: Dict[str, Any]
    ) -> List[Triple]:
        """Extract relations between entities."""
        logger.info(f"Extracting relations from {len(entities)} entities")
        # TODO: Implement relation extraction logic
        return []
