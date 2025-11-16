#!/usr/bin/env python3
"""
Test vLLM deployment with NeuraLog.

Tests that vLLM is correctly configured and can generate responses.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuralog import Engine
from neuralog.core.config import Config


def test_vllm_basic():
    """Test basic vLLM generation."""
    print("Testing vLLM Basic Generation")
    print("="*60)

    # Load production config
    config = Config.from_yaml(Path("configs/production_h200.yaml"))

    print(f"\nModel: {config.llm.model}")
    print(f"Provider: {config.llm.provider}")
    print(f"Tensor Parallel Size: {config.llm.tensor_parallel_size}")
    print(f"GPU Memory Utilization: {config.llm.gpu_memory_utilization}")

    # Initialize engine
    print("\nInitializing NeuraLog engine with vLLM...")
    engine = Engine(config)

    # Test simple generation
    print("\nTesting generation...")
    prompt = "What is 2+2? Answer in one sentence."

    response = engine.llm_interface.generate(
        prompt=prompt,
        temperature=0.1,
        max_tokens=100
    )

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

    print("\n✓ vLLM test passed!")


def test_vllm_extraction():
    """Test extraction with vLLM."""
    print("\n\nTesting vLLM Extraction")
    print("="*60)

    config = Config.from_yaml(Path("configs/production_h200.yaml"))
    engine = Engine(config)

    text = """
    Dr. Sarah Chen is a researcher at Stanford University. She works on
    artificial intelligence and machine learning. Her recent paper on
    neural-symbolic AI was published in Nature.
    """

    print(f"\nExtracting from text:\n{text}")

    # Use semantic workspace
    kg = engine.extract_with_semantic_workspace(
        text=text,
        workspace_name="test_vllm"
    )

    print(f"\n✓ Extraction complete!")
    print(f"  Entities: {len(kg.entities)}")
    print(f"  Relations: {len(kg.triples)}")

    for i, entity in enumerate(list(kg.entities.values())[:5], 1):
        print(f"  {i}. {entity.label}")


def test_vllm_server():
    """Test vLLM server mode."""
    print("\n\nTesting vLLM Server Mode")
    print("="*60)

    config = Config.from_yaml(Path("configs/production_vllm_server.yaml"))

    print(f"\nServer URL: {config.llm.base_url}")
    print(f"Model: {config.llm.model}")

    try:
        engine = Engine(config)

        response = engine.llm_interface.generate(
            prompt="Hello, respond in one sentence.",
            temperature=0.1,
            max_tokens=50
        )

        print(f"\nResponse: {response}")
        print("\n✓ vLLM server test passed!")

    except Exception as e:
        print(f"\n⚠️  Server not running: {e}")
        print("Start server with: ./scripts/start_vllm_server.sh")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM deployment")
    parser.add_argument(
        "--mode",
        choices=["basic", "extraction", "server", "all"],
        default="basic",
        help="Test mode"
    )

    args = parser.parse_args()

    try:
        if args.mode in ["basic", "all"]:
            test_vllm_basic()

        if args.mode in ["extraction", "all"]:
            test_vllm_extraction()

        if args.mode in ["server", "all"]:
            test_vllm_server()

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
