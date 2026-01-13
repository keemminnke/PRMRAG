# Installation Guide

## Quick Start

```bash
pip install -r requirements.txt
```

## Known Issues & Solutions

### vLLM + flash_attn + numpy Compatibility

**Problem**: vLLM 0.13.0 requires specific versions of numpy, pandas, and pyarrow to work with flash_attn 2.4.x.

**Symptoms**:
```
ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Solution** (already in requirements.txt):
```bash
pip install "numpy<2.0,>=1.24.0" "pandas>=2.0.0,<2.4.0" "pyarrow>=22.0.0,<23.0.0"
```

### If Installation Fails

**Step 1**: Clean install
```bash
pip uninstall numpy pandas pyarrow vllm flash-attn -y
pip install -r requirements.txt
```

**Step 2**: If pyarrow/pandas still cause issues
```bash
pip install --upgrade --force-reinstall "numpy<2" pyarrow pandas
```

**Step 3**: Verify installation
```bash
python -c "import numpy, pandas, pyarrow, vllm; print(f'numpy: {numpy.__version__}, vllm: {vllm.__version__}')"
```

Expected output:
```
numpy: 1.26.x, vllm: 0.13.x
```

## Tested Configurations

### Working Configuration (2026-01-13)
- **Python**: 3.10
- **CUDA**: 12.x
- **torch**: 2.9.0
- **vllm**: 0.13.0
- **flash-attn**: 2.4.2
- **numpy**: 1.26.4 (< 2.0 is CRITICAL!)
- **pandas**: 2.3.3
- **pyarrow**: 22.0.0

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (A100/H100 recommended)
- **VRAM**: Minimum 24GB for Qwen2.5-7B
- **RAM**: Minimum 32GB
- **Disk**: ~50GB for models and data

## Development Installation

```bash
# Clone repository
git clone <repo-url>
cd PRMRAG

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

## Troubleshooting

### flash_attn compilation issues
If flash_attn fails to install:
```bash
# Option 1: Use PyTorch SDPA backend (slower but works)
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

# Option 2: Skip flash_attn (not recommended for production)
pip install vllm --no-deps
pip install <other dependencies>
```

### CUDA version mismatch
```bash
# Check CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Additional Notes

- **flash-attn**: Installation takes 10-15 minutes (compiles from source)
- **vLLM**: Uses ~14GB VRAM for Qwen2.5-7B
- **Embeddings**: BGE-M3 cache is ~20GB for KILT Wikipedia
- **BM25 index**: ~2GB for KILT Wikipedia

## Verification

Run the test suite:
```bash
# Quick test (mock components)
python /tmp/test_batch_parallel_real.py

# Full test (requires data)
python scripts/batch_test_hybrid.py --num-questions 3 --num-rollouts 4
```
