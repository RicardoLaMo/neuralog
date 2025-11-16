# Production Deployment Guide

This guide covers deploying NeuraLog with vLLM for high-performance inference on NVIDIA H200 GPUs.

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Installation](#installation)
4. [Model Selection](#model-selection)
5. [Configuration](#configuration)
6. [Deployment Modes](#deployment-modes)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Overview

NeuraLog supports production deployment with vLLM, providing:

- **High-throughput inference**: 2-4x faster than standard transformers
- **Memory efficiency**: PagedAttention reduces memory usage by 50%
- **Multi-GPU support**: Tensor parallelism for large models (671B)
- **Prefix caching**: Reuse KV cache for common prompts
- **OpenAI-compatible API**: Drop-in replacement for OpenAI endpoints

### Supported Models

| Model | Parameters | GPUs Required | Context Length | Best For |
|-------|-----------|---------------|----------------|----------|
| Qwen/QwQ-32B-Preview | 32B | 1x H200 | 32K | Reasoning, step-by-step analysis |
| DeepSeek-R1 | 671B | 4x H200 | 16K | Advanced reasoning, large-scale |
| Qwen2.5-72B-Instruct | 72B | 2x H200 | 128K | Long context, general purpose |
| Meta-Llama-3.1-70B | 70B | 2x H200 | 128K | General purpose, instruction following |

---

## Hardware Requirements

### Minimum Requirements

- **GPU**: 1x NVIDIA H200 (141GB HBM3e)
- **RAM**: 256GB system memory
- **Storage**: 200GB SSD for model weights
- **CUDA**: 12.1 or higher

### Recommended for Production

- **GPU**: 4x NVIDIA H200 for large models
- **RAM**: 512GB ECC memory
- **Storage**: 1TB NVMe SSD
- **Network**: 100Gbps for distributed training
- **CUDA**: 12.4 with latest drivers

### Model-Specific Requirements

**Qwen/QwQ-32B-Preview** (Single H200):
- GPU Memory: ~60GB (90% utilization of 141GB)
- System RAM: 128GB
- Storage: 65GB for model weights

**DeepSeek R1** (4x H200):
- GPU Memory: ~520GB total (95% utilization across 4 GPUs)
- System RAM: 512GB
- Storage: 1.3TB for model weights

---

## Installation

### Step 1: Base Installation

```bash
# Clone repository
git clone https://github.com/yourusername/neuralog.git
cd neuralog

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install core dependencies
pip install -e .
```

### Step 2: Production Dependencies

```bash
# Install vLLM and production dependencies
pip install -e ".[production]"

# Verify installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Step 3: Optional Dependencies

```bash
# For graph neural networks (if using graph embeddings)
pip install -e ".[graph]"

# For LangChain integration (if needed)
pip install -e ".[langchain]"

# Install all optional dependencies
pip install -e ".[all]"
```

### Step 4: Verify GPU Setup

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check vLLM GPU detection
python -c "from vllm import LLM; print('vLLM GPU ready')"
```

---

## Model Selection

### Decision Matrix

Choose model based on your use case:

#### Qwen/QwQ-32B-Preview ✅ **Recommended for Most Users**

**Pros:**
- Fits on single H200 (cost-effective)
- Excellent reasoning capabilities
- Long context support (32K tokens)
- Fast inference (~100 tokens/sec)

**Cons:**
- Smaller than frontier models
- May struggle with very complex reasoning

**Use cases:**
- Financial compliance reasoning
- Policy analysis and extraction
- Knowledge graph construction
- General-purpose applications

**Configuration:** `configs/production_qwen_qwq.yaml`

#### DeepSeek R1 (671B)

**Pros:**
- State-of-the-art reasoning
- Handles extremely complex tasks
- Large knowledge capacity

**Cons:**
- Requires 4x H200 GPUs (expensive)
- Slower inference (~25 tokens/sec)
- Large storage requirements (1.3TB)

**Use cases:**
- Enterprise-scale deployments
- Research requiring frontier capabilities
- Complex multi-hop reasoning
- When cost is not primary concern

**Configuration:** `configs/production_deepseek_r1.yaml`

#### Model Size vs Performance Trade-off

```
Cost-Effectiveness:  QwQ-32B > Qwen2.5-72B > Llama-3.1-70B > DeepSeek-R1
Reasoning Quality:   DeepSeek-R1 > QwQ-32B > Qwen2.5-72B > Llama-3.1-70B
Inference Speed:     QwQ-32B > Qwen2.5-72B > Llama-3.1-70B > DeepSeek-R1
```

---

## Configuration

### Configuration Files

NeuraLog uses YAML configuration files in `configs/`:

```
configs/
├── production_h200.yaml           # Default: QwQ-32B on single H200
├── production_qwen_qwq.yaml       # QwQ-32B optimized
├── production_deepseek_r1.yaml    # DeepSeek R1 on 4x H200
└── production_vllm_server.yaml    # Client mode for vLLM server
```

### Key Configuration Parameters

#### LLM Configuration

```yaml
llm:
  provider: "vllm"                    # or "vllm-server"
  model: "Qwen/QwQ-32B-Preview"       # HuggingFace model ID

  # Inference parameters
  temperature: 0.1                     # Lower = more deterministic
  max_tokens: 4096                     # Max generation length
  top_p: 0.95                          # Nucleus sampling

  # vLLM-specific settings
  tensor_parallel_size: 1              # Number of GPUs for model
  gpu_memory_utilization: 0.90         # GPU memory to use (0.0-1.0)
  max_model_len: 32768                 # Max context length
  trust_remote_code: true              # For custom models
  dtype: "bfloat16"                    # Data type (bfloat16/float16)

  # Performance optimizations
  enable_prefix_caching: true          # Cache common prompts
  use_v2_block_manager: true           # New memory manager
  max_num_seqs: 256                    # Batch size
```

#### Single GPU vs Multi-GPU

**Single H200** (QwQ-32B):
```yaml
llm:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90        # Conservative for safety
  max_num_seqs: 256
```

**Multi-GPU** (DeepSeek R1):
```yaml
llm:
  tensor_parallel_size: 4             # Split across 4 GPUs
  pipeline_parallel_size: 1           # No pipeline parallelism
  gpu_memory_utilization: 0.95        # Higher with multiple GPUs
  max_num_seqs: 512                   # Larger batch
  distributed_executor_backend: "ray" # Ray for multi-GPU
```

#### Ontology Configuration

```yaml
ontology:
  reasoner: "elk"                     # ELK reasoner (fast)
  enable_reasoning: true              # Enable OWL reasoning
```

#### Embedding Configuration

```yaml
embedding:
  model: "sentence-transformers/all-mpnet-base-v2"
  device: "cuda"                      # or "cuda:0" for specific GPU
  batch_size: 128                     # Embedding batch size
```

#### Verification Configuration

```yaml
verification:
  enable_verification: true           # Z3 formal verification
  solver: "z3"                        # SMT solver
```

---

## Deployment Modes

### Mode 1: Direct vLLM (Recommended for Single Application)

Load vLLM engine directly in Python process.

**Pros:**
- Lowest latency
- No network overhead
- Simple setup

**Cons:**
- Single application only
- Model loaded in-process

**Usage:**

```python
from pathlib import Path
from neuralog import Engine
from neuralog.core.config import Config

# Load configuration
config = Config.from_yaml(Path("configs/production_qwen_qwq.yaml"))

# Initialize engine (loads vLLM)
engine = Engine(config)

# Extract knowledge
text = """
Dr. Sarah Chen is a researcher at Stanford University.
She works on artificial intelligence and published in Nature.
"""

kg = engine.extract_with_semantic_workspace(
    text=text,
    workspace_name="research"
)

print(f"Extracted {len(kg.entities)} entities")
for entity in list(kg.entities.values())[:5]:
    print(f"  - {entity.label} ({entity.entity_type})")
```

**Configuration:** Use `configs/production_qwen_qwq.yaml` or `configs/production_deepseek_r1.yaml`

### Mode 2: vLLM Server (Recommended for Multiple Applications)

Run vLLM as OpenAI-compatible API server.

**Pros:**
- Serve multiple applications
- OpenAI-compatible API
- Easy integration
- Separate process isolation

**Cons:**
- Network latency (~5-10ms)
- Requires server management

#### Starting the Server

```bash
# Start server with QwQ-32B on single H200
./scripts/start_vllm_server.sh Qwen/QwQ-32B-Preview 1

# Start server with DeepSeek R1 on 4 H200s
./scripts/start_vllm_server.sh deepseek-ai/DeepSeek-R1 4

# Custom port and settings
./scripts/start_vllm_server.sh Qwen/QwQ-32B-Preview 1 8080 0.90 32768
```

Server starts at `http://localhost:8000` with endpoints:
- `GET /v1/models` - List available models
- `POST /v1/completions` - Generate completions
- `POST /v1/chat/completions` - Chat completions

#### Using the Server

```python
from pathlib import Path
from neuralog import Engine
from neuralog.core.config import Config

# Load server configuration
config = Config.from_yaml(Path("configs/production_vllm_server.yaml"))

# Engine connects to server
engine = Engine(config)

# Use normally
response = engine.llm_interface.generate(
    prompt="Explain neurosymbolic AI",
    temperature=0.1,
    max_tokens=500
)
print(response)
```

**Configuration:** Use `configs/production_vllm_server.yaml`

#### Server Configuration

Edit `configs/production_vllm_server.yaml`:

```yaml
llm:
  provider: "vllm-server"
  model: "Qwen/QwQ-32B-Preview"  # Must match server model
  base_url: "http://localhost:8000/v1"

  temperature: 0.1
  max_tokens: 4096
  top_p: 0.95
```

---

## Performance Tuning

### GPU Memory Utilization

Balance between throughput and stability:

```yaml
# Conservative (safer, lower throughput)
gpu_memory_utilization: 0.85

# Balanced (recommended)
gpu_memory_utilization: 0.90

# Aggressive (higher throughput, may OOM)
gpu_memory_utilization: 0.95
```

**Guideline:**
- Single GPU: Start at 0.90, decrease if OOM errors
- Multi-GPU: Can use 0.95 (more memory headroom)
- Long contexts: Use 0.85-0.88 (reserve for KV cache)

### Batch Size Tuning

Adjust `max_num_seqs` for throughput:

```yaml
# Small batch (low latency)
max_num_seqs: 64

# Medium batch (balanced)
max_num_seqs: 256

# Large batch (high throughput)
max_num_seqs: 512
```

**Trade-offs:**
- **Smaller batch**: Lower latency, lower throughput
- **Larger batch**: Higher latency, higher throughput

**Recommendation:** Start with 256, adjust based on workload.

### Prefix Caching

Enable for repeated prompts:

```yaml
llm:
  enable_prefix_caching: true
```

**Benefits:**
- Reuses KV cache for common prompt prefixes
- 2-3x speedup for repeated system prompts
- Essential for extraction tasks with fixed templates

**When to use:**
- Knowledge extraction (fixed extraction prompts)
- Policy analysis (repeated regulation templates)
- Chat applications (system message reuse)

### Context Length vs Memory

Longer context = more GPU memory:

```yaml
# Short context (more memory for batch)
max_model_len: 8192

# Medium context (balanced)
max_model_len: 16384

# Long context (less memory for batch)
max_model_len: 32768
```

**Memory usage:**
- 8K context: ~40GB for QwQ-32B
- 16K context: ~50GB for QwQ-32B
- 32K context: ~60GB for QwQ-32B

### CUDA Graphs

Enable for faster inference:

```yaml
llm:
  enforce_eager: false  # Use CUDA graphs (faster)
```

**Benefits:**
- 10-20% faster inference
- Lower latency for short generations

**Disable if:**
- Variable-length inputs
- Dynamic control flow
- Debugging

### Multi-GPU Optimization

For DeepSeek R1 (4x H200):

```yaml
llm:
  tensor_parallel_size: 4
  distributed_executor_backend: "ray"
  max_parallel_loading_workers: 4  # Parallel model loading
```

**Tips:**
- Use `ray` backend for 2+ GPUs
- Set `max_parallel_loading_workers` = number of GPUs
- Enable high-bandwidth interconnect (NVLink)

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Reduce GPU memory utilization:
   ```yaml
   gpu_memory_utilization: 0.85  # from 0.90
   ```

2. Decrease batch size:
   ```yaml
   max_num_seqs: 128  # from 256
   ```

3. Reduce context length:
   ```yaml
   max_model_len: 16384  # from 32768
   ```

4. Use smaller model:
   - Switch from DeepSeek R1 to QwQ-32B
   - Use quantized model (int8/int4)

#### 2. Slow Inference

**Symptoms:**
- <10 tokens/second
- High latency (>5 seconds)

**Solutions:**

1. Enable CUDA graphs:
   ```yaml
   enforce_eager: false
   ```

2. Enable prefix caching:
   ```yaml
   enable_prefix_caching: true
   ```

3. Use flash attention:
   ```bash
   pip install flash-attn>=2.5.0
   ```

4. Increase batch size:
   ```yaml
   max_num_seqs: 512  # Process more in parallel
   ```

5. Check GPU utilization:
   ```bash
   nvidia-smi dmon -s u
   ```
   Should be 90%+ during inference.

#### 3. Model Loading Failures

**Symptoms:**
```
ValueError: trust_remote_code must be True
```

**Solutions:**

1. Enable trust_remote_code:
   ```yaml
   trust_remote_code: true
   ```

2. Update transformers:
   ```bash
   pip install --upgrade transformers>=4.40.0
   ```

3. Clear cache and retry:
   ```bash
   rm -rf ~/.cache/huggingface/hub
   ```

#### 4. Import Errors

**Symptoms:**
```
ImportError: cannot import name 'LLM' from 'vllm'
```

**Solutions:**

1. Reinstall vLLM:
   ```bash
   pip uninstall vllm -y
   pip install vllm>=0.4.0 --no-cache-dir
   ```

2. Check CUDA compatibility:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
   Should match CUDA toolkit version.

3. Install from source (if binary fails):
   ```bash
   pip install git+https://github.com/vllm-project/vllm.git
   ```

#### 5. Version Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solutions:**

1. Use exact versions from pyproject.toml:
   ```bash
   pip install "numpy>=1.24.0,<2.0.0" "torch>=2.1.0,<2.5.0"
   ```

2. Install in clean environment:
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install -e ".[production]"
   ```

3. Check conflicting packages:
   ```bash
   pip check
   ```

#### 6. Multi-GPU Not Working

**Symptoms:**
- Only 1 GPU utilized
- `tensor_parallel_size=4` but `nvidia-smi` shows 1 GPU

**Solutions:**

1. Set CUDA_VISIBLE_DEVICES:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

2. Install Ray:
   ```bash
   pip install ray>=2.9.0
   ```

3. Use Ray backend:
   ```yaml
   distributed_executor_backend: "ray"
   ```

4. Check GPU topology:
   ```bash
   nvidia-smi topo -m
   ```

### Debugging Tips

#### Enable Verbose Logging

```yaml
log_level: "DEBUG"
```

#### Check vLLM Logs

vLLM logs show detailed initialization and inference info:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Profile GPU Usage

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Detailed profiling
nsys profile -o profile.qdrep python your_script.py
```

#### Test vLLM Directly

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/QwQ-32B-Preview",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    trust_remote_code=True
)

outputs = llm.generate(
    ["Explain AI in one sentence."],
    SamplingParams(temperature=0.1, max_tokens=100)
)

print(outputs[0].outputs[0].text)
```

---

## Testing

### Quick Test

```bash
# Test basic generation
python scripts/test_vllm.py --mode basic

# Test extraction
python scripts/test_vllm.py --mode extraction

# Test server mode
python scripts/test_vllm.py --mode server

# Test all
python scripts/test_vllm.py --mode all
```

### Benchmark Performance

```python
import time
from neuralog import Engine
from neuralog.core.config import Config

config = Config.from_yaml("configs/production_qwen_qwq.yaml")
engine = Engine(config)

# Warmup
engine.llm_interface.generate("Test", max_tokens=10)

# Benchmark
prompts = ["Explain AI"] * 100
start = time.time()

for prompt in prompts:
    engine.llm_interface.generate(prompt, max_tokens=100)

elapsed = time.time() - start
throughput = len(prompts) / elapsed

print(f"Throughput: {throughput:.2f} requests/sec")
print(f"Latency: {elapsed/len(prompts)*1000:.2f} ms/request")
```

---

## Production Checklist

Before deploying to production:

- [ ] Install production dependencies: `pip install -e ".[production]"`
- [ ] Choose appropriate model for use case (QwQ-32B recommended)
- [ ] Configure GPU memory utilization (start at 0.90)
- [ ] Enable prefix caching for repeated prompts
- [ ] Set appropriate batch size (`max_num_seqs`)
- [ ] Configure context length based on needs
- [ ] Test with `scripts/test_vllm.py`
- [ ] Benchmark performance on representative workload
- [ ] Set up monitoring (GPU usage, latency, throughput)
- [ ] Configure error handling and retries
- [ ] Document deployment configuration
- [ ] Set up backup/failover if critical

---

## Additional Resources

- **vLLM Documentation**: https://docs.vllm.ai/
- **Model Hub**: https://huggingface.co/models
- **NVIDIA H200 Specs**: https://www.nvidia.com/en-us/data-center/h200/
- **NeuraLog Architecture**: `ARCHITECTURE.md`
- **Semantic Layer Guide**: `SEMANTIC_LAYER.md`

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review vLLM logs (set `log_level: "DEBUG"`)
3. Open GitHub issue with configuration and error logs
