#!/usr/bin/env python3
"""
Biomedical Information Extraction Example

Demonstrates using NeuraLog with a biomedical ontology for extracting
disease-gene relationships from scientific text.
"""

from pathlib import Path
from neuralog import Engine
from neuralog.core.config import Config
from neuralog.utils.logger import setup_logger


def main():
    """Run biomedical extraction example."""

    setup_logger(log_level="INFO")

    # Initialize with configuration
    config = Config()

    # You can load a custom config file:
    # config = Config.from_yaml(Path("configs/biomedical.yaml"))

    engine = Engine(config)

    # In a real scenario, you would load a biomedical ontology
    # ontology_path = Path("ontologies/mondo.owl")  # Disease ontology
    # engine.load_ontology(ontology_path)

    # Sample biomedical text
    text = """
    Recent studies have shown that mutations in the BRCA1 gene are associated
    with increased risk of breast cancer and ovarian cancer. BRCA1 encodes a
    tumor suppressor protein that plays a critical role in DNA repair. Patients
    with BRCA1 mutations have a 60-70% lifetime risk of developing breast cancer.
    The protein interacts with RAD51 to facilitate homologous recombination repair.
    """

    print("Biomedical Information Extraction")
    print("="*60)
    print(f"\nText:\n{text}\n")

    try:
        # Extract with verification enabled
        print("Extracting knowledge with neurosymbolic validation...")
        result = engine.extract_knowledge(
            text=text,
            verify=True,  # Enable formal verification
            confidence_threshold=0.7
        )

        print(f"\n{'='*60}")
        print("RESULTS")
        print('='*60)

        print(f"\nExtracted {len(result.triples)} high-confidence triples")

        # Group triples by relation type
        by_relation = {}
        for triple in result.triples:
            pred = triple.predicate.uri if hasattr(triple.predicate, 'uri') else str(triple.predicate)
            if pred not in by_relation:
                by_relation[pred] = []
            by_relation[pred].append(triple)

        for relation, triples in by_relation.items():
            print(f"\n{relation}: ({len(triples)} triples)")
            for triple in triples[:5]:
                subj = triple.subject.label if hasattr(triple.subject, 'label') else str(triple.subject)
                obj = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                status = "✓ VERIFIED" if triple.confidence_level.value == "verified" else f"({triple.confidence:.2f})"
                print(f"  - {subj} → {obj} {status}")

        if result.verification_status:
            print(f"\nVerification: {result.verification_status}")

        # Build knowledge graph
        kg = result.to_knowledge_graph("brca1_kg")

        # Generate embeddings
        print("\nGenerating graph embeddings...")
        try:
            embeddings = engine.build_embeddings(kg, method="rdf_walk")
            print(f"Generated embeddings for {len(embeddings)} entities")

            # Find similar entities
            if "BRCA1" in [e.label for e in kg.entities.values()]:
                brca1_entity = [e for e in kg.entities.values() if e.label == "BRCA1"][0]
                similar = engine.graph_embedder.find_similar_entities(
                    embeddings,
                    brca1_entity.uri,
                    top_k=5
                )
                print(f"\nEntities similar to BRCA1:")
                for uri, score in similar:
                    entity = kg.entities.get(uri)
                    if entity:
                        print(f"  - {entity.label}: {score:.3f}")
        except NotImplementedError:
            print("(Embedding generation requires graph dependencies)")

        # Demonstrate querying (when implemented)
        # query = "What genes are associated with breast cancer?"
        # results = engine.query(query, kg=kg)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires:")
        print("- LLM API key (NEURALOG_LLM_API_KEY)")
        print("- Optional: Biomedical ontology (e.g., MONDO, GO)")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
