#!/usr/bin/env python3
"""
Simple Information Extraction Example

Demonstrates basic usage of NeuraLog for extracting knowledge from text.
"""

from pathlib import Path
from neuralog import Engine
from neuralog.core.config import Config
from neuralog.utils.logger import setup_logger


def main():
    """Run simple extraction example."""

    # Setup logging
    setup_logger(log_level="INFO")

    # Load configuration
    config = Config()

    # For this example, we'll use environment variables or defaults
    # Make sure to set NEURALOG_LLM_API_KEY in your environment

    # Initialize engine
    print("Initializing NeuraLog engine...")
    engine = Engine(config)

    # Sample text
    text = """
    Albert Einstein was a German-born theoretical physicist who developed the
    theory of relativity. He received the Nobel Prize in Physics in 1921 for
    his explanation of the photoelectric effect. Einstein worked at the
    Institute for Advanced Study in Princeton, New Jersey.
    """

    print(f"\nExtracting knowledge from text:\n{text}\n")

    # Extract knowledge (without formal verification for this simple example)
    try:
        result = engine.extract_knowledge(
            text=text,
            verify=False,  # Disable verification for faster results
            confidence_threshold=0.6
        )

        # Display results
        print(f"\n{'='*60}")
        print("EXTRACTION RESULTS")
        print('='*60)

        print(f"\nEntities found: {len(result.entities)}")
        for entity in result.entities[:10]:  # Show first 10
            print(f"  - {entity.label} ({entity.uri})")
            if entity.attributes.get('ontology_type'):
                print(f"    Type: {entity.attributes['ontology_type']}")

        print(f"\nRelations found: {len(result.relations)}")
        for relation in result.relations[:10]:
            print(f"  - {relation.label} ({relation.uri})")

        print(f"\nTriples extracted: {len(result.triples)}")
        for triple in result.triples[:10]:
            subj = triple.subject.label if hasattr(triple.subject, 'label') else str(triple.subject)
            pred = triple.predicate.uri if hasattr(triple.predicate, 'uri') else str(triple.predicate)
            obj = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
            print(f"  - ({subj}, {pred}, {obj}) [confidence: {triple.confidence:.2f}]")

        print(f"\nOverall confidence: {result.confidence:.2f}")
        print(f"Confidence level: {result.confidence_level.value}")

        # Convert to knowledge graph
        kg = result.to_knowledge_graph("einstein_kg")
        print(f"\nKnowledge graph created: {kg.name}")
        print(f"  - {len(kg.entities)} entities")
        print(f"  - {len(kg.relations)} relation types")
        print(f"  - {len(kg.triples)} triples")

    except Exception as e:
        print(f"\nError during extraction: {e}")
        print("\nMake sure to:")
        print("1. Set NEURALOG_LLM_API_KEY in your environment")
        print("2. Install required dependencies: pip install -e .")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
