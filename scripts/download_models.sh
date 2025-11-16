#!/bin/bash
#
# NeuraLog Model Download Script
# Downloads models from HuggingFace Hub using huggingface-cli
#
# Usage:
#   ./scripts/download_models.sh <model_name> [options]
#
# Examples:
#   ./scripts/download_models.sh qwen-qwq-32b
#   ./scripts/download_models.sh deepseek-r1 --token YOUR_HF_TOKEN
#   ./scripts/download_models.sh all-llm
#   ./scripts/download_models.sh all-embeddings
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface/hub}"
HF_TOKEN="${HF_TOKEN:-}"
CONCURRENT_DOWNLOADS=4

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print usage
usage() {
    cat << EOF
NeuraLog Model Download Script

Usage: $0 <model_name> [options]

Model Names (LLMs):
  qwen-qwq-32b           Qwen/QwQ-32B-Preview (Recommended)
  deepseek-r1            deepseek-ai/DeepSeek-R1 (671B, requires 4x H200)
  qwen-2.5-72b           Qwen/Qwen2.5-72B-Instruct
  llama-3.1-70b          meta-llama/Meta-Llama-3.1-70B-Instruct
  qwen-2.5-7b            Qwen/Qwen2.5-7B-Instruct (for testing)
  mistral-7b             mistralai/Mistral-7B-Instruct-v0.3 (for testing)
  all-llm                Download all LLM models
  all-small              Download all small models (7B)

Model Names (Embeddings):
  mpnet                  all-mpnet-base-v2 (Default)
  minilm                 all-MiniLM-L6-v2 (Fast)
  gte-large              thenlper/gte-large
  bge-large              BAAI/bge-large-en-v1.5
  e5-large               intfloat/e5-large-v2
  all-embeddings         Download all embedding models

Options:
  --token TOKEN          HuggingFace token (required for gated models like Llama)
  --cache-dir DIR        Cache directory (default: ~/.cache/huggingface/hub)
  --no-verify            Skip free space verification
  --help                 Show this help message

Environment Variables:
  HF_TOKEN               HuggingFace token
  HF_HOME                HuggingFace cache directory

Examples:
  # Download recommended model (QwQ-32B)
  $0 qwen-qwq-32b

  # Download DeepSeek R1 with token
  $0 deepseek-r1 --token hf_xxxxx

  # Download all embedding models
  $0 all-embeddings

  # Download with custom cache directory
  $0 qwen-qwq-32b --cache-dir /data/models

EOF
    exit 1
}

# Check if huggingface-cli is installed
check_hf_cli() {
    if ! command -v huggingface-cli &> /dev/null; then
        print_msg "$RED" "Error: huggingface-cli not found!"
        echo ""
        echo "Install with:"
        echo "  pip install -U huggingface_hub[cli]"
        echo ""
        exit 1
    fi
}

# Check disk space
check_disk_space() {
    local required_gb=$1
    local available_kb=$(df "$CACHE_DIR" | tail -1 | awk '{print $4}')
    local available_gb=$((available_kb / 1024 / 1024))

    print_msg "$BLUE" "Required space: ${required_gb}GB"
    print_msg "$BLUE" "Available space: ${available_gb}GB"

    if [ "$available_gb" -lt "$required_gb" ]; then
        print_msg "$RED" "Error: Insufficient disk space!"
        print_msg "$YELLOW" "Required: ${required_gb}GB, Available: ${available_gb}GB"
        exit 1
    fi
}

# Login to HuggingFace
hf_login() {
    if [ -n "$HF_TOKEN" ]; then
        print_msg "$BLUE" "Logging in to HuggingFace Hub..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    else
        print_msg "$YELLOW" "No HF_TOKEN provided. Skipping login."
        print_msg "$YELLOW" "Note: Some models (like Llama) require authentication."
    fi
}

# Download model using huggingface-cli
download_model() {
    local repo_id=$1
    local repo_type=${2:-"model"}  # model or dataset
    local required_space=${3:-100}  # GB

    print_msg "$GREEN" "=========================================="
    print_msg "$GREEN" "Downloading: $repo_id"
    print_msg "$GREEN" "=========================================="

    # Check disk space
    if [ "$VERIFY_SPACE" = "true" ]; then
        check_disk_space "$required_space"
    fi

    # Download command
    print_msg "$BLUE" "Starting download..."
    huggingface-cli download "$repo_id" \
        --repo-type "$repo_type" \
        --cache-dir "$CACHE_DIR" \
        --resume-download \
        --local-dir-use-symlinks False

    if [ $? -eq 0 ]; then
        print_msg "$GREEN" "✓ Successfully downloaded: $repo_id"
    else
        print_msg "$RED" "✗ Failed to download: $repo_id"
        return 1
    fi
}

# Download specific model by name
download_by_name() {
    local model_name=$1

    case "$model_name" in
        # LLM Models
        qwen-qwq-32b)
            download_model "Qwen/QwQ-32B-Preview" "model" 70
            ;;
        deepseek-r1)
            print_msg "$YELLOW" "Warning: DeepSeek R1 is very large (~1.3TB)"
            print_msg "$YELLOW" "This download may take several hours or days."
            read -p "Continue? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                download_model "deepseek-ai/DeepSeek-R1" "model" 1400
            fi
            ;;
        qwen-2.5-72b)
            download_model "Qwen/Qwen2.5-72B-Instruct" "model" 150
            ;;
        llama-3.1-70b)
            if [ -z "$HF_TOKEN" ]; then
                print_msg "$RED" "Error: Llama models require HuggingFace token"
                print_msg "$YELLOW" "Get token from: https://huggingface.co/settings/tokens"
                print_msg "$YELLOW" "Usage: $0 llama-3.1-70b --token YOUR_TOKEN"
                exit 1
            fi
            print_msg "$YELLOW" "Note: Requires accepting Meta's license agreement"
            print_msg "$YELLOW" "Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct"
            download_model "meta-llama/Meta-Llama-3.1-70B-Instruct" "model" 150
            ;;
        qwen-2.5-7b)
            download_model "Qwen/Qwen2.5-7B-Instruct" "model" 15
            ;;
        mistral-7b)
            download_model "mistralai/Mistral-7B-Instruct-v0.3" "model" 15
            ;;

        # Embedding Models
        mpnet)
            download_model "sentence-transformers/all-mpnet-base-v2" "model" 1
            ;;
        minilm)
            download_model "sentence-transformers/all-MiniLM-L6-v2" "model" 1
            ;;
        gte-large)
            download_model "thenlper/gte-large" "model" 2
            ;;
        bge-large)
            download_model "BAAI/bge-large-en-v1.5" "model" 2
            ;;
        e5-large)
            download_model "intfloat/e5-large-v2" "model" 2
            ;;

        # Batch downloads
        all-llm)
            print_msg "$YELLOW" "Downloading all LLM models (this will take a long time and ~1.7TB)"
            read -p "Continue? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                download_by_name qwen-qwq-32b
                download_by_name qwen-2.5-72b
                download_by_name qwen-2.5-7b
                download_by_name mistral-7b
                print_msg "$YELLOW" "Skipping DeepSeek R1 (too large). Download separately if needed."
            fi
            ;;
        all-small)
            download_by_name qwen-2.5-7b
            download_by_name mistral-7b
            ;;
        all-embeddings)
            download_by_name mpnet
            download_by_name minilm
            download_by_name gte-large
            download_by_name bge-large
            download_by_name e5-large
            ;;

        *)
            print_msg "$RED" "Error: Unknown model name: $model_name"
            echo ""
            usage
            ;;
    esac
}

# Main script
main() {
    # Parse arguments
    MODEL_NAME=""
    VERIFY_SPACE=true

    while [[ $# -gt 0 ]]; do
        case $1 in
            --token)
                HF_TOKEN="$2"
                shift 2
                ;;
            --cache-dir)
                CACHE_DIR="$2"
                shift 2
                ;;
            --no-verify)
                VERIFY_SPACE=false
                shift
                ;;
            --help|-h)
                usage
                ;;
            *)
                if [ -z "$MODEL_NAME" ]; then
                    MODEL_NAME="$1"
                else
                    print_msg "$RED" "Error: Unknown option: $1"
                    usage
                fi
                shift
                ;;
        esac
    done

    if [ -z "$MODEL_NAME" ]; then
        print_msg "$RED" "Error: Model name required"
        echo ""
        usage
    fi

    # Check prerequisites
    check_hf_cli

    # Create cache directory
    mkdir -p "$CACHE_DIR"

    print_msg "$BLUE" "NeuraLog Model Download"
    print_msg "$BLUE" "======================="
    print_msg "$BLUE" "Cache directory: $CACHE_DIR"
    print_msg "$BLUE" "Model: $MODEL_NAME"
    echo ""

    # Login if token provided
    hf_login

    # Download model
    download_by_name "$MODEL_NAME"

    print_msg "$GREEN" ""
    print_msg "$GREEN" "=========================================="
    print_msg "$GREEN" "Download complete!"
    print_msg "$GREEN" "=========================================="
    print_msg "$BLUE" "Models are cached at: $CACHE_DIR"
    print_msg "$BLUE" ""
    print_msg "$BLUE" "Next steps:"
    print_msg "$BLUE" "  1. Update your config to use the model"
    print_msg "$BLUE" "  2. Run: python -m neuralog --config configs/production_*.yaml"
    print_msg "$BLUE" ""
}

# Run main function
main "$@"
