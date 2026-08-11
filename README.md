<div align="center">

```text
 _     _     __  __ ____                      TF5
| |   | |   |  \/  / ___|___  _ __ ___  _ __  
| |   | |   | |\/| | |   / _ \| '_ ` _ \| '_ \ 
| |___| |___| |  | | |__| (_) | | | | | | |_) |
|_____|_____|_|  |_|\____\___/|_| |_| |_| .__/ 
                                        |_|    
```

**llm-compressor — Transformers 5.x Compatibility**

*Enabling quantization of modern architectures by bridging llm-compressor and Transformers 5.0+*

[![Framework: llm-compressor](https://img.shields.io/badge/Framework-llm--compressor-blue?style=for-the-badge)](https://github.com/vllm-project/llm-compressor)
[![Language: Python](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## 📑 Table of Contents

- [⚡ Overview](#-overview)
- [🎯 The Problem](#-the-problem)
- [✨ Features & Patches](#-features--patches)
- [📦 Installation & Usage](#-installation--usage)
- [🔧 MTP Weight Preservation](#-mtp-weight-preservation)
- [📄 License](#-license)

---

## ⚡ Overview

Provides patches for [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) to work seamlessly with `transformers >= 5.0`. This makes it possible to quantize new model architectures like Qwen3.5 which depend on the newer transformers releases.

> [!NOTE]  
> **Verified Compatibility:** Tested against all 126 Python modules in llm-compressor 0.10.x with transformers 5.5.3. Zero import failures after patches.

---

## 🎯 The Problem

`llm-compressor 0.10.x` strictly pins `transformers<=4.57.6`. However, Qwen3.5 (`qwen3_5` model type) is only available in transformers 5.0+. Only **2 imports** actually break when upgrading:

1. `TORCH_INIT_FUNCTIONS` removed from `transformers.modeling_utils` in 5.x.
2. `Conv1D` moved from `transformers.modeling_utils` to `transformers.pytorch_utils`.

---

## ✨ Features & Patches

| File | Fix |
|---|---|
| `patches/dev.py.patch` | 3-tier import: `transformers.initialization` → `modeling_utils` → build from `torch.nn.init` |
| `patches/module.py.patch` | Try `pytorch_utils` first, fall back to `modeling_utils` |

---

## 📦 Installation & Usage

**1. Apply the patches:**
```bash
python apply_patches.py                          # auto-detect site-packages
python apply_patches.py /path/to/site-packages   # explicit path
python apply_patches.py --dry-run                 # preview only
```

**2. Install transformers 5.x:**
```bash
pip install "transformers>=5.5" --no-deps
```

**3. Full Quantization Example:**
```bash
python quantize_qwen35.py \
    --model huihui-ai/Huihui-Qwen3.5-27B-abliterated \
    --output ./Qwen3.5-27B-W4A16 \
    --scheme W4A16
```

---

## 🔧 MTP Weight Preservation

Qwen3.5's MTP (Multi-Token Prediction) weights are silently dropped during standard quantization because:
`Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`

The provided `inject_mtp_weights.py` script copies them back post-quantization:

```bash
python inject_mtp_weights.py --source /path/to/original --quantized /path/to/quantized
```

---

## 📄 License

Apache-2.0 (same as llm-compressor)
