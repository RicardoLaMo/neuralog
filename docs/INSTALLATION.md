# NeuraLog Installation Guide

Complete installation guide for NeuraLog with different deployment scenarios.

## Table of Contents

1. [Quick Install](#quick-install)
2. [Installation Options](#installation-options)
3. [Requirements Files](#requirements-files)
4. [Model Download](#model-download)
5. [Environment Setup](#environment-setup)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Quick Install

### For Development

```bash
# Clone repository
git clone https://github.com/yourusername/neuralog.git
cd neuralog

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install with pip
pip install -e .

# Or install with requirements.txt
pip install -r requirements.txt
```

### For Production (with vLLM)

```bash
# Install core + production dependencies
pip install -r requirements.txt -r requirements-production.txt

# Or use pyproject.toml
pip install -e ".[production]"

# Download recommended model
./scripts/download_models.sh qwen-qwq-32b
```

---

## Installation Options

### Option 1: Using `pyproject.toml` (Recommended)

NeuraLog uses modern Python packaging with optional dependency groups.

```bash
# Install core dependencies only
pip install -e .

# Install with production dependencies (vLLM)
pip install -e ".[production]"

# Install with development tools
pip install -e ".[dev]"

# Install with graph neural networks
pip install -e ".[graph]"

# Install everything
pip install -e ".[all]"
```

**Dependency Groups:**

- **Core**: Essential dependencies (transformers, DeepOnto, z3-solver, etc.)
- **production**: vLLM, Ray, Triton, Flash Attention for H200 GPU
- **dev**: Testing, linting, documentation tools
- **graph**: Graph neural networks (PyG, DGL)
- **langchain**: LangChain integration
- **all**: Everything above

### Option 2: Using Requirements Files

For environments that don't support modern pyproject.toml:

```bash
# Core dependencies
pip install -r requirements.txt

# Production deployment
pip install -r requirements.txt -r requirements-production.txt

# Development
pip install -r requirements.txt -r requirements-dev.txt

# Graph neural networks
pip install -r requirements.txt -r requirements-graph.txt

# Everything
pip install -r requirements.txt \
            -r requirements-production.txt \
            -r requirements-dev.txt \
            -r requirements-graph.txt
```

---

## Requirements Files

### `requirements.txt` - Core Dependencies

Essential dependencies for basic functionality:

- **Deep Learning**: PyTorch, Transformers, Accelerate
- **Ontology**: DeepOnto, OWLReady2, RDFLib
- **Embeddings**: Sentence-Transformers, Gensim
- **Verification**: Z3 SMT Solver
- **LLM APIs**: OpenAI, Anthropic
- **Utilities**: Pydantic, YAML, Redis, FAISS

**Install:**
```bash
pip install -r requirements.txt
```

### `requirements-production.txt` - Production/vLLM

High-performance inference for H200 GPU deployment:

- **vLLM**: High-throughput inference engine
- **Ray**: Distributed execution for multi-GPU
- **Triton**: Optimized CUDA kernels
- **xformers**: Memory-efficient attention
- **Flash Attention**: Fast attention implementation
- **FAISS-GPU**: GPU-accelerated vector search
- **Monitoring**: Prometheus, profiling tools

**Install:**
```bash
pip install -r requirements.txt -r requirements-production.txt
```

### `requirements-dev.txt` - Development Tools

Tools for development, testing, and code quality:

- **Testing**: pytest, pytest-cov, hypothesis
- **Code Quality**: black, ruff, mypy, pylint
- **Documentation**: Sphinx, RTD theme
- **Development**: pre-commit, ipdb, Jupyter

**Install:**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### `requirements-graph.txt` - Graph Neural Networks

Optional dependencies for graph neural network features:

- **PyTorch Geometric**: Graph neural networks
- **DGL**: Deep Graph Library
- **Graph Tools**: igraph, graph-tool
- **Embeddings**: node2vec, karateclub
- **Visualization**: matplotlib, seaborn, plotly

**Install:**
```bash
pip install -r requirements.txt -r requirements-graph.txt
```

---

## Model Download

NeuraLog provides two methods for downloading models from HuggingFace Hub.

### Method 1: Bash Script (Recommended for Interactive Use)

```bash
# Download recommended model (QwQ-32B)
./scripts/download_models.sh qwen-qwq-32b

# Download with HuggingFace token (for gated models)
./scripts/download_models.sh llama-3.1-70b --token hf_xxxxx

# Download all embedding models
./scripts/download_models.sh all-embeddings

# See all options
./scripts/download_models.sh --help
```

**Available Models:**

**LLM Models:**
- `qwen-qwq-32b` - Qwen/QwQ-32B-Preview (Recommended) ⭐
- `deepseek-r1` - DeepSeek-R1 (671B, requires 4x H200)
- `qwen-2.5-72b` - Qwen2.5-72B-Instruct (long context)
- `llama-3.1-70b` - Llama-3.1-70B-Instruct (requires token)
- `qwen-2.5-7b` - Qwen2.5-7B-Instruct (for testing)
- `mistral-7b` - Mistral-7B-Instruct (for testing)

**Embedding Models:**
- `mpnet` - all-mpnet-base-v2 (Default)
- `minilm` - all-MiniLM-L6-v2 (Fast)
- `gte-large` - GTE-Large
- `bge-large` - BGE-Large-EN
- `e5-large` - E5-Large-v2

**Batch Downloads:**
- `all-llm` - All LLM models (excludes DeepSeek R1)
- `all-small` - Small models for testing (7B)
- `all-embeddings` - All embedding models

### Method 2: Python Script (Recommended for Automation)

```bash
# List available models
python scripts/manage_models.py list

# Show detailed model information
python scripts/manage_models.py info qwen-qwq-32b

# Download a model
python scripts/manage_models.py download qwen-qwq-32b

# Download with token
python scripts/manage_models.py download llama-3.1-70b --token hf_xxxxx

# Verify model is accessible
python scripts/manage_models.py verify qwen-qwq-32b

# Help
python scripts/manage_models.py --help
```

### Method 3: HuggingFace CLI (Direct)

Install HuggingFace CLI:
```bash
pip install -U huggingface_hub[cli]
```

Download models:
```bash
# Login (for gated models)
huggingface-cli login

# Download model
huggingface-cli download Qwen/QwQ-32B-Preview

# Download to specific directory
huggingface-cli download Qwen/QwQ-32B-Preview \
    --cache-dir /data/models \
    --resume-download

# List local models
huggingface-cli scan-cache
```

### Model Storage

Models are cached at:
- **Linux/Mac**: `~/.cache/huggingface/hub`
- **Windows**: `C:\Users\USERNAME\.cache\huggingface\hub`
- **Custom**: Set `HF_HOME` environment variable

```bash
# Set custom cache directory
export HF_HOME=/data/huggingface
export TRANSFORMERS_CACHE=/data/huggingface/transformers
```

---

## Environment Setup

### Step 1: Python Environment

**Requirements:**
- Python 3.11 or higher
- pip 23.0+
- virtualenv or conda

**Create virtual environment:**

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Or using conda
conda create -n neuralog python=3.11
conda activate neuralog
```

### Step 2: System Dependencies

#### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt-get update

# Install Java (required for DeepOnto/OWLAPI)
sudo apt-get install -y openjdk-11-jdk

# Install system libraries
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget

# Verify Java
java -version
```

#### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Java
brew install openjdk@11

# Link Java
sudo ln -sfn /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk \
    /Library/Java/JavaVirtualMachines/openjdk-11.jdk
```

#### Windows

1. Install Java 11+: https://adoptium.net/
2. Install Build Tools: https://visualstudio.microsoft.com/downloads/ (Build Tools for Visual Studio)
3. Install Git: https://git-scm.com/download/win

### Step 3: GPU Setup (For Production)

#### NVIDIA CUDA Toolkit

**Check GPU:**
```bash
nvidia-smi
```

**Install CUDA 12.1+ (for H200):**

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

**Install cuDNN:**
```bash
sudo apt-get install -y libcudnn8 libcudnn8-dev
```

### Step 4: Configuration Files

#### Create `.env` file

```bash
# Copy example
cp .env.example .env

# Edit with your API keys
nano .env
```

**`.env` contents:**
```bash
# LLM Provider Configuration
NEURALOG_LLM_PROVIDER="vllm"  # or "openai", "anthropic"
NEURALOG_LLM_MODEL="Qwen/QwQ-32B-Preview"

# API Keys (if using cloud providers)
NEURALOG_LLM_API_KEY="your-api-key-here"
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."

# HuggingFace Token (for gated models)
HF_TOKEN="hf_..."

# Cache Directories
HF_HOME="/data/huggingface"
TRANSFORMERS_CACHE="/data/huggingface/transformers"

# Redis (optional, for distributed caching)
NEURALOG_REDIS_URL="redis://localhost:6379/0"

# Logging
NEURALOG_LOG_LEVEL="INFO"
```

#### Select Configuration File

Choose appropriate config for your deployment:

```bash
# Single H200 with QwQ-32B (recommended)
export NEURALOG_CONFIG="configs/production_qwen_qwq.yaml"

# 4x H200 with DeepSeek R1
export NEURALOG_CONFIG="configs/production_deepseek_r1.yaml"

# vLLM server mode
export NEURALOG_CONFIG="configs/production_vllm_server.yaml"
```

---

## Verification

### Step 1: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.11+

# Check Java
java -version  # Should be 11+

# Check CUDA (if using GPU)
nvidia-smi
nvcc --version

# Check installed packages
pip list | grep -E "torch|transformers|vllm|deeponto"
```

### Step 2: Test Basic Import

```python
# test_imports.py
import sys

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers: {e}")

try:
    import deeponto
    print(f"✅ DeepOnto: {deeponto.__version__}")
except Exception as e:
    print(f"❌ DeepOnto: {e}")

try:
    import vllm
    print(f"✅ vLLM: {vllm.__version__}")
except Exception as e:
    print(f"⚠️  vLLM: {e} (optional for production)")

try:
    import z3
    print(f"✅ Z3: {z3.get_version_string()}")
except Exception as e:
    print(f"❌ Z3: {e}")

print("\n✅ All core imports successful!")
```

Run:
```bash
python test_imports.py
```

### Step 3: Test NeuraLog

```bash
# Test basic functionality
python scripts/test_vllm.py --mode basic

# Or use Python
python -c "from neuralog import Engine; print('✅ NeuraLog ready!')"
```

### Step 4: Run Example

```bash
# Simple extraction example
python examples/simple_extraction.py

# Financial compliance demo (Jupyter)
jupyter notebook examples/financial_compliance_demo.ipynb
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error: No module named 'XXX'

**Problem:**
```
ImportError: No module named 'torch'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. Java Not Found

**Problem:**
```
RuntimeError: Java not found
```

**Solution:**
```bash
# Install Java 11+
sudo apt-get install openjdk-11-jdk  # Linux
brew install openjdk@11               # macOS

# Verify
java -version

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  # Linux
export JAVA_HOME=/opt/homebrew/opt/openjdk@11         # macOS
```

#### 3. CUDA Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Reduce GPU memory utilization in config
# Edit configs/production_*.yaml
gpu_memory_utilization: 0.85  # Reduce from 0.90

# Or use smaller batch size
max_num_seqs: 128  # Reduce from 256

# Or use smaller model
./scripts/download_models.sh qwen-2.5-7b
```

#### 4. Version Conflicts

**Problem:**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Solution:**
```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate

# Install with specific versions
pip install "numpy>=1.24.0,<2.0.0" "torch>=2.1.0,<2.5.0"
pip install -r requirements.txt

# Or use pip-tools for resolution
pip install pip-tools
pip-compile requirements.txt
pip-sync requirements.txt
```

#### 5. HuggingFace Download Fails

**Problem:**
```
HfHubHTTPError: 401 Client Error: Unauthorized
```

**Solution:**
```bash
# Get HuggingFace token from: https://huggingface.co/settings/tokens

# Set token
export HF_TOKEN="hf_..."

# Or login
huggingface-cli login

# Download with token
./scripts/download_models.sh llama-3.1-70b --token $HF_TOKEN
```

#### 6. vLLM Installation Fails

**Problem:**
```
ERROR: Failed building wheel for vllm
```

**Solution:**
```bash
# Ensure CUDA is installed
nvcc --version

# Install build dependencies
pip install --upgrade pip setuptools wheel ninja

# Install from source
pip install git+https://github.com/vllm-project/vllm.git

# Or use conda
conda install -c conda-forge vllm
```

### Getting Help

If you encounter issues not covered here:

1. **Check logs**: Set `NEURALOG_LOG_LEVEL=DEBUG` for detailed output
2. **Check documentation**:
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System design
   - [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - Production guide
   - [SEMANTIC_LAYER.md](SEMANTIC_LAYER.md) - Semantic layer
3. **Search issues**: https://github.com/yourusername/neuralog/issues
4. **Open new issue**: Provide error logs, system info, and steps to reproduce

---

## Next Steps

After successful installation:

1. **Read documentation**:
   - [Quick Start](../README.md#quick-start)
   - [Architecture Overview](ARCHITECTURE.md)
   - [Production Deployment](PRODUCTION_DEPLOYMENT.md)

2. **Download models**:
   ```bash
   ./scripts/download_models.sh qwen-qwq-32b
   ```

3. **Run examples**:
   ```bash
   python examples/simple_extraction.py
   jupyter notebook examples/financial_compliance_demo.ipynb
   ```

4. **Configure for your use case**:
   - Edit `configs/production_*.yaml`
   - Set environment variables in `.env`
   - Review [Model Selection Guide](PRODUCTION_DEPLOYMENT.md#model-selection)

Happy coding! 🚀
