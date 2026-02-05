# Fix: Avoid Re-downloading Models in Notebook

## Problem

The notebook downloads models from ModelScope to `/workspace/models/`, but the pre-downloaded models from the model downloader DaemonSet are in `/root/.cache/huggingface/` (mapped from `/mnt/models`).

## Solution

The models are **already available** in the container! The pod mounts `/mnt/models` to `/root/.cache/huggingface/` (see `k8s/pods.py` lines 111-115).

You just need to tell DiffSynth-Studio to use HuggingFace instead of ModelScope.

## Quick Fix for the Notebook

### Option 1: Use HuggingFace Cache (Recommended)

Replace cell 8 in the notebook with:

```python
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# IMPORTANT: Remove this line to use HuggingFace cache instead of ModelScope
# os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"

# Use HuggingFace cache (models already downloaded)
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
from PIL import Image
import pandas as pd
import numpy as np

# Models will be loaded from /root/.cache/huggingface (already downloaded!)
qwen_image = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
qwen_image.enable_lora_magic()
```

### Option 2: Set Environment at Notebook Start

Add this as the FIRST cell in the notebook:

```python
import os

# Use pre-downloaded HuggingFace models
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"

# Don't download from ModelScope
if "MODELSCOPE_DOMAIN" in os.environ:
    del os.environ["MODELSCOPE_DOMAIN"]

print("✅ Using HuggingFace cache at /root/.cache/huggingface")
print("✅ Models are already downloaded - no re-download needed!")
```

Then keep the rest of the notebook as-is.

## Why This Works

1. **Model downloader** downloads models to `/mnt/models` on the host
2. **Jupyter pod** mounts `/mnt/models` to `/root/.cache/huggingface` in the container
3. **HuggingFace libraries** automatically check `/root/.cache/huggingface` for models
4. **No re-download** happens because models are already there!

## Verify Models Are Available

Run this in a notebook cell to verify:

```python
import os
from pathlib import Path

hf_cache = Path("/root/.cache/huggingface/hub")
if hf_cache.exists():
    models = list(hf_cache.glob("models--*"))
    print(f"✅ Found {len(models)} models in cache:")
    for model in models[:10]:  # Show first 10
        print(f"  - {model.name}")
else:
    print("❌ HuggingFace cache not found")
```

## Models That Should Be Pre-downloaded

Based on `model_download_qwen_image_workshop.yaml`, these models are already downloaded:

1. ✅ `Qwen/Qwen-Image` (base model)
2. ✅ `Qwen/Qwen-Image-Edit` (edit model)
3. ✅ `Qwen/Qwen-Image-Edit-2509` (multi-image edit)
4. ✅ `DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1` (LoRA)
5. ✅ `DiffSynth-Studio/Qwen-Image-Edit-F2P` (LoRA)
6. ✅ `Artiprocher/dataset_dog` (dataset)

## Summary

**No code changes needed in `k8s/pods.py`** - the volume mount is already correct!

**Just remove** `os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"` from the notebook and the models will be loaded from cache automatically.
