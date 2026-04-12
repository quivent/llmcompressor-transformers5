# llm-compressor Transformers 5.x Compatibility Patches

Patches for [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) to support `transformers>=5.0` and preserve Qwen3.5 MTP (Multi-Token Prediction) weights during quantization.

## Problem

Two issues block llm-compressor from working with transformers 5.x:

1. **`TORCH_INIT_FUNCTIONS` removed from `transformers.modeling_utils`** — The fallback to `{}` makes `skip_weights_initialize()` a no-op, causing random weight initialization to corrupt calibration data during quantization.

2. **`Conv1D` import path changed** — In older transformers it lived in `modeling_utils`, newer versions moved it to `pytorch_utils`.

Additionally, when quantizing Qwen3.5 models:

3. **MTP weights are silently dropped** — `Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` causes `from_pretrained()` to never load MTP tensors, so `save_pretrained()` never saves them.

## Patches

| File | Change |
|------|--------|
| `llmcompressor/utils/dev.py` | 3-tier import for `TORCH_INIT_FUNCTIONS`: try `transformers.initialization`, then `transformers.modeling_utils`, then build from `torch.nn.init` |
| `llmcompressor/utils/pytorch/module.py` | Try `transformers.pytorch_utils.Conv1D` before `transformers.modeling_utils.Conv1D` |
| `setup.cfg` / `pyproject.toml` | Relax pin from `transformers<=4.57.6` to `transformers>=4.56.1,<6.0` |

## Usage

### Apply patches to an existing installation

```bash
python apply_patches.py /path/to/site-packages
```

This modifies llmcompressor in-place and verifies imports afterward.

### Quantize Qwen3.5 with MTP preservation

```bash
python quantize_qwen35.py \
    --model Qwen/Qwen3.5-27B \
    --output ./Qwen3.5-27B-FP8 \
    --scheme FP8 \
    --num-calibration-samples 512
```

### Inject MTP weights (standalone)

If you already have a quantized model and just need to add the MTP weights back:

```bash
python inject_mtp_weights.py \
    --source /path/to/Qwen3.5-27B \
    --quantized /path/to/Qwen3.5-27B-FP8
```

### Serve with vLLM

```bash
vllm serve ./Qwen3.5-27B-FP8 \
    --num-speculative-tokens 1 \
    --trust-remote-code
```

## Version pin recommendation

In `setup.cfg` or `pyproject.toml`, change:

```diff
- transformers<=4.57.6
+ transformers>=4.56.1,<6.0
```

## Files

```
.
├── README.md                  # This file
├── apply_patches.py           # Automated patch applier with verification
├── inject_mtp_weights.py      # Post-quantization MTP weight injector
├── quantize_qwen35.py         # Complete quantization example
└── patches/
    ├── dev.py.patch           # Unified diff for utils/dev.py
    └── module.py.patch        # Unified diff for utils/pytorch/module.py
```

## How it works

### TORCH_INIT_FUNCTIONS fallback

The critical insight: an empty dict `{}` is not a safe fallback. `skip_weights_initialize()` iterates over `TORCH_INIT_FUNCTIONS.keys()` to patch `torch.nn.init` functions with no-ops during model loading. Without this, random initialization runs on every tensor, corrupting calibration.

The robust fallback builds the dict from `torch.nn.init` directly:

```python
TORCH_INIT_FUNCTIONS = {
    name: getattr(torch.nn.init, name)
    for name in dir(torch.nn.init)
    if callable(getattr(torch.nn.init, name))
    and name.endswith("_")
    and not name.startswith("__")
}
```

### MTP weight injection

Qwen3.5 models have MTP head weights prefixed with `mtp` (e.g., `mtp.0.model.layers.0.self_attn.q_proj.weight`). The model class explicitly ignores these during loading. The injector:

1. Reads all `^mtp` tensors from the original model safetensors
2. Writes them into a dedicated `mtp_weights.safetensors` shard
3. Updates `model.safetensors.index.json` with the new weight map
4. Copies `mtp_num_hidden_layers` into the output `config.json`
