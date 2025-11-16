#!/usr/bin/env python3
"""
Semantic Workspace Demo - Graph Dynamics vs. Chunking

Demonstrates the new semantic approach using:
- Generative Semantic Workspace (arxiv:2511.07587)
- Episodic Transformer Memory
- Graph dynamics for entity evolution

This replaces naive chunking with intelligent semantic processing.
"""

from pathlib import Path
from neuralog import Engine
from neuralog.core.config import Config
from neuralog.utils.logger import setup_logger


def main():
    """Run semantic workspace demonstration."""

    setup_logger(log_level="INFO")

    print("="*70)
    print("SEMANTIC WORKSPACE DEMO: Graph Dynamics vs. Chunking")
    print("="*70)
    print()

    # Initialize engine
    config = Config()
    engine = Engine(config)

    # Narrative text with temporal structure and entity evolution
    # This demonstrates why semantic processing is better than chunking
    text = """
    Dr. Sarah Chen began her career as a junior researcher at MIT in 2015.
    She focused on machine learning applications in healthcare.

    In 2017, Sarah published her groundbreaking paper on neural networks
    for medical diagnosis. The work caught the attention of Stanford
    University, where she was offered a faculty position.

    After moving to Stanford in 2018, Dr. Chen expanded her research to
    include symbolic AI methods. She collaborated with Prof. Robert Kim
    from the Philosophy Department on interpretable AI systems.

    By 2020, their collaboration resulted in a new framework combining
    neural and symbolic approaches. The framework was called NeuroSymb
    and attracted significant research funding.

    In 2021, Dr. Chen became director of the AI Safety Institute at Stanford.
    Under her leadership, the institute grew from 5 researchers to 25.
    She also began advising government agencies on AI policy.

    Today, Dr. Chen splits her time between research, teaching, and policy work.
    Her work has influenced how AI systems are developed and deployed worldwide.
    """

    print("Input narrative with temporal evolution")
    print("-"*70)
    print(text[:300] + "...")
    print()

    # OLD APPROACH: Chunking (for comparison)
    print("\n" + "="*70)
    print("OLD APPROACH: Naive Chunking")
    print("="*70)
    print("Problems with chunking:")
    print("- Breaks narrative flow across chunks")
    print("- Loses temporal structure")
    print("- Misses entity evolution (Sarah's career progression)")
    print("- No coherence between chunks")
    print()

    # NEW APPROACH: Semantic Workspace
    print("="*70)
    print("NEW APPROACH: Semantic Workspace with Graph Dynamics")
    print("="*70)
    print()

    try:
        # Extract using semantic workspace
        print("Extracting knowledge with semantic workspace...")
        kg = engine.extract_with_semantic_workspace(
            text=text,
            workspace_name="sarah_chen_timeline",
            use_episodic_memory=True
        )

        print(f"\n✓ Extraction complete!")
        print(f"  - Entities: {len(kg.entities)}")
        print(f"  - Relations: {len(kg.triples)}")
        print()

        # Show entity evolution (graph dynamics)
        print("-"*70)
        print("ENTITY EVOLUTION: Dr. Sarah Chen")
        print("-"*70)

        # Find Sarah's entity
        sarah_entity = None
        for entity in kg.entities.values():
            if "sarah" in entity.label.lower() or "chen" in entity.label.lower():
                sarah_entity = entity
                break

        if sarah_entity:
            # Get trajectory from graph dynamics
            trajectory = engine.graph_dynamics.get_entity_trajectory(sarah_entity.uri)

            print(f"\nTracked {len(trajectory)} state changes for Dr. Chen:")
            for i, state in enumerate(trajectory, 1):
                print(f"\n{i}. Position {state.narrative_position}:")
                print(f"   Time: {state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                if state.location:
                    print(f"   Location: {state.location}")
                print(f"   Properties: {list(state.properties.keys())}")
                if state.properties:
                    for key, value in list(state.properties.items())[:3]:
                        print(f"     - {key}: {value}")

        # Show temporal graph snapshots
        print("\n" + "-"*70)
        print("TEMPORAL GRAPH SNAPSHOTS")
        print("-"*70)

        # Get graph at different narrative positions
        if 'graph_dynamics' in kg.metadata:
            dynamics = kg.metadata['graph_dynamics']
            total_positions = dynamics.get('current_position', 0)

            if total_positions > 0:
                print(f"\nGraph evolved across {total_positions} narrative positions")

                # Show metrics at beginning, middle, end
                for position in [0, total_positions // 2, total_positions - 1]:
                    metrics = engine.graph_dynamics.compute_graph_metrics(position)
                    print(f"\nPosition {position}:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"  - {key}: {value:.3f}")
                        else:
                            print(f"  - {key}: {value}")

        # Show episodic memory statistics
        print("\n" + "-"*70)
        print("EPISODIC MEMORY")
        print("-"*70)

        if 'episodic_memory' in kg.metadata:
            mem_stats = kg.metadata['episodic_memory']
            print(f"\nMemory Statistics:")
            for key, value in mem_stats.items():
                print(f"  - {key}: {value}")

            # Get context for query
            print("\nQuerying episodic memory:")
            context = engine.episodic_memory.get_context(
                query="What research did Sarah do?",
                top_k=3
            )

            print(f"Found {len(context)} relevant memory states")
            for i, state in enumerate(context, 1):
                print(f"\n  {i}. Timestep {state.timestep}")
                print(f"     Observation: {str(state.observation)[:100]}...")

        # Compare with workspace statistics
        print("\n" + "-"*70)
        print("SEMANTIC WORKSPACE STATISTICS")
        print("-"*70)

        workspace_stats = engine.semantic_workspace.get_statistics()
        print("\nWorkspace State:")
        for key, value in workspace_stats.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.3f}")
            else:
                print(f"  - {key}: {value}")

        # Query workspace narratively
        print("\n" + "-"*70)
        print("NARRATIVE QUERYING")
        print("-"*70)

        print("\nQuerying: Events in first half of narrative")
        early_events = engine.semantic_workspace.query_narrative(
            query="early career",
            temporal_range=(0, workspace_stats.get('narrative_length', 10) // 2)
        )

        print(f"Found {len(early_events)} early events:")
        for event in early_events[:3]:
            print(f"\n  - Position {event.narrative_position}: {event.event_type}")
            print(f"    Entities: {[e.label for e in event.entities[:3]]}")
            print(f"    Coherence: {event.coherence_score:.2f}")

        print("\n" + "="*70)
        print("KEY ADVANTAGES OF SEMANTIC WORKSPACE")
        print("="*70)
        print()
        print("✓ Maintains narrative coherence across entire document")
        print("✓ Tracks entity evolution through time")
        print("✓ Preserves temporal and spatial relationships")
        print("✓ Enables temporal querying (state at time T)")
        print("✓ Builds coherent event sequences, not isolated chunks")
        print("✓ Uses episodic memory for long-range context")
        print("✓ 51% more token-efficient than traditional RAG (per paper)")
        print("✓ 20% better performance on episodic benchmarks")
        print()

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires:")
        print("- LLM API key (NEURALOG_LLM_API_KEY)")
        print("- Dependencies: pip install -e .")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
