#!/usr/bin/env python3
"""
NeuraLog Model Management Script

Provides programmatic model download and management using HuggingFace Hub API.

Usage:
    python scripts/manage_models.py download qwen-qwq-32b
    python scripts/manage_models.py list
    python scripts/manage_models.py info qwen-qwq-32b
    python scripts/manage_models.py verify qwen-qwq-32b
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

try:
    from huggingface_hub import snapshot_download, list_repo_files, model_info
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


# Model mapping
MODEL_MAPPING = {
    # LLM Models
    "qwen-qwq-32b": "Qwen/QwQ-32B-Preview",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    # Embedding Models
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "gte-large": "thenlper/gte-large",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-large": "intfloat/e5-large-v2",
}


def load_model_config() -> Dict:
    """Load model configuration from configs/models.yaml"""
    config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_cache_dir() -> Path:
    """Get HuggingFace cache directory"""
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if not cache_dir:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return Path(cache_dir)


def get_model_repo_id(model_name: str) -> str:
    """Get HuggingFace repo ID from model name"""
    return MODEL_MAPPING.get(model_name, model_name)


def download_model(
    model_name: str,
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    resume: bool = True,
) -> Path:
    """
    Download model from HuggingFace Hub

    Args:
        model_name: Model name or HuggingFace repo ID
        token: HuggingFace token (for gated models)
        cache_dir: Cache directory (default: HF_HOME or ~/.cache/huggingface)
        resume: Resume interrupted downloads

    Returns:
        Path to downloaded model
    """
    repo_id = get_model_repo_id(model_name)

    if cache_dir is None:
        cache_dir = get_cache_dir()

    print(f"📥 Downloading: {repo_id}")
    print(f"📁 Cache directory: {cache_dir}")

    try:
        # Get model info first
        info = model_info(repo_id, token=token)
        print(f"ℹ️  Model info:")
        print(f"   - ID: {info.id}")
        print(f"   - Downloads: {info.downloads:,}")
        print(f"   - Likes: {info.likes}")

        # Check size if available
        if hasattr(info, 'siblings') and info.siblings:
            total_size = sum(f.size for f in info.siblings if hasattr(f, 'size') and f.size)
            print(f"   - Size: {total_size / 1024**3:.2f} GB")

        # Download model
        print("\n🚀 Starting download...")
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            resume_download=resume,
            token=token,
            local_files_only=False,
        )

        print(f"✅ Successfully downloaded to: {local_path}")
        return Path(local_path)

    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            print("❌ Error: Authentication required")
            print("   Get token from: https://huggingface.co/settings/tokens")
            print(f"   Usage: python {sys.argv[0]} download {model_name} --token YOUR_TOKEN")
        elif e.response.status_code == 403:
            print("❌ Error: Access forbidden")
            print(f"   You may need to accept the model license at:")
            print(f"   https://huggingface.co/{repo_id}")
        else:
            print(f"❌ Error downloading model: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def list_models(verbose: bool = False) -> None:
    """List available models"""
    config = load_model_config()

    print("📚 Available Models\n")

    # LLM Models
    print("🤖 Large Language Models:")
    print("-" * 80)
    if "llm_models" in config:
        for key, model in config["llm_models"].items():
            name = model.get("name", "Unknown")
            params = model.get("parameters", 0)
            gpus = model.get("recommended_gpus", "?")

            # Format parameters
            if params >= 1_000_000_000:
                params_str = f"{params / 1_000_000_000:.0f}B"
            else:
                params_str = f"{params / 1_000_000:.0f}M"

            print(f"  {key:20s} - {name:45s} ({params_str}, {gpus} GPU)")

            if verbose:
                features = model.get("features", [])
                recommended_for = model.get("recommended_for", [])
                print(f"      Features: {', '.join(features)}")
                print(f"      Best for: {', '.join(recommended_for)}")
                print()

    print()

    # Embedding Models
    print("📊 Embedding Models:")
    print("-" * 80)
    if "embedding_models" in config:
        for key, model in config["embedding_models"].items():
            name = model.get("name", "Unknown")
            dims = model.get("dimensions", "?")
            performance = model.get("performance", "")

            print(f"  {key:20s} - {name:45s} ({dims}D, {performance})")

            if verbose:
                recommended_for = model.get("recommended_for", [])
                print(f"      Best for: {', '.join(recommended_for)}")
                print()

    print()
    print("💡 Download a model with:")
    print("   python scripts/manage_models.py download <model_name>")
    print("   ./scripts/download_models.sh <model_name>")


def show_model_info(model_name: str) -> None:
    """Show detailed information about a model"""
    config = load_model_config()
    repo_id = get_model_repo_id(model_name)

    # Find model in config
    model_config = None
    model_type = None

    if "llm_models" in config:
        for key, model in config["llm_models"].items():
            if key == model_name or model.get("name") == repo_id:
                model_config = model
                model_type = "LLM"
                break

    if not model_config and "embedding_models" in config:
        for key, model in config["embedding_models"].items():
            if key == model_name or model.get("name") == repo_id:
                model_config = model
                model_type = "Embedding"
                break

    print(f"📋 Model Information: {model_name}")
    print("=" * 80)

    if model_config:
        print(f"Type: {model_type}")
        print(f"HuggingFace ID: {model_config.get('name', repo_id)}")

        if model_type == "LLM":
            params = model_config.get("parameters", 0)
            params_str = f"{params / 1_000_000_000:.0f}B" if params >= 1_000_000_000 else f"{params / 1_000_000:.0f}M"

            print(f"Parameters: {params_str}")
            print(f"Context Length: {model_config.get('context_length', 'Unknown'):,} tokens")
            print(f"Recommended GPUs: {model_config.get('recommended_gpus', 'Unknown')} x {model_config.get('gpu_type', 'Unknown')}")
            print(f"GPU Memory: {model_config.get('gpu_memory_required', 'Unknown')}")
            print(f"Data Type: {model_config.get('dtype', 'Unknown')}")
            print(f"Throughput: {model_config.get('throughput_estimate', 'Unknown')}")
            print(f"Cost Effectiveness: {model_config.get('cost_effectiveness', 'Unknown')}")
            print(f"Reasoning Quality: {model_config.get('reasoning_quality', 'Unknown')}")

            features = model_config.get("features", [])
            if features:
                print(f"\nFeatures:")
                for feature in features:
                    print(f"  - {feature}")

            recommended_for = model_config.get("recommended_for", [])
            if recommended_for:
                print(f"\nRecommended For:")
                for use_case in recommended_for:
                    print(f"  - {use_case}")

            config_file = model_config.get("config_file")
            if config_file:
                print(f"\nConfiguration File: configs/{config_file}")

            notes = model_config.get("notes")
            if notes:
                print(f"\n⚠️  Notes: {notes}")

        else:  # Embedding model
            print(f"Dimensions: {model_config.get('dimensions', 'Unknown')}")
            print(f"Max Sequence Length: {model_config.get('max_sequence_length', 'Unknown')}")
            print(f"Model Size: {model_config.get('model_size', 'Unknown')}")
            print(f"Performance: {model_config.get('performance', 'Unknown')}")

            recommended_for = model_config.get("recommended_for", [])
            if recommended_for:
                print(f"\nRecommended For:")
                for use_case in recommended_for:
                    print(f"  - {use_case}")

            notes = model_config.get("notes")
            if notes:
                print(f"\n⚠️  Notes: {notes}")
    else:
        print(f"HuggingFace ID: {repo_id}")
        print("\nNo detailed configuration found in configs/models.yaml")

    print("\n" + "=" * 80)
    print("💡 Download this model with:")
    print(f"   python scripts/manage_models.py download {model_name}")
    print(f"   ./scripts/download_models.sh {model_name}")


def verify_model(model_name: str, token: Optional[str] = None) -> None:
    """Verify if model is downloaded and accessible"""
    repo_id = get_model_repo_id(model_name)
    cache_dir = get_cache_dir()

    print(f"🔍 Verifying model: {model_name}")
    print(f"   Repo ID: {repo_id}")
    print(f"   Cache directory: {cache_dir}")

    try:
        # Try to get model info
        info = model_info(repo_id, token=token)
        print(f"✅ Model exists on HuggingFace Hub")
        print(f"   Downloads: {info.downloads:,}")
        print(f"   Likes: {info.likes}")

        # Check if downloaded locally
        # This is a simplified check - actual location depends on HF cache structure
        model_cache = cache_dir / f"models--{repo_id.replace('/', '--')}"

        if model_cache.exists():
            print(f"✅ Model appears to be cached locally at:")
            print(f"   {model_cache}")
        else:
            print(f"⚠️  Model not found in local cache")
            print(f"   Download with: python scripts/manage_models.py download {model_name}")

    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            print("❌ Authentication required")
            print("   Add --token argument with your HuggingFace token")
        elif e.response.status_code == 403:
            print("❌ Access forbidden - you may need to accept model license")
        else:
            print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="NeuraLog Model Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python scripts/manage_models.py list

  # Show detailed model information
  python scripts/manage_models.py info qwen-qwq-32b

  # Download a model
  python scripts/manage_models.py download qwen-qwq-32b

  # Download with HuggingFace token (for gated models)
  python scripts/manage_models.py download llama-3.1-70b --token hf_xxxxx

  # Verify model is accessible
  python scripts/manage_models.py verify qwen-qwq-32b
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed information"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model_name", help="Model name")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_name", help="Model name")
    download_parser.add_argument(
        "--token",
        help="HuggingFace token (required for gated models)",
        default=os.environ.get("HF_TOKEN")
    )
    download_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory (default: HF_HOME or ~/.cache/huggingface)"
    )
    download_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume interrupted downloads"
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify model accessibility")
    verify_parser.add_argument("model_name", help="Model name")
    verify_parser.add_argument(
        "--token",
        help="HuggingFace token",
        default=os.environ.get("HF_TOKEN")
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "list":
        list_models(verbose=args.verbose)
    elif args.command == "info":
        show_model_info(args.model_name)
    elif args.command == "download":
        download_model(
            args.model_name,
            token=args.token,
            cache_dir=args.cache_dir,
            resume=not args.no_resume
        )
    elif args.command == "verify":
        verify_model(args.model_name, token=args.token)


if __name__ == "__main__":
    main()
