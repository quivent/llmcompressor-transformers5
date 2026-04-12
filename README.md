# llm-compressor — Transformers 5.x Compatibility

Patches for [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) to work with `transformers >= 5.0`, enabling quantization of new model architectures like Qwen3.5.

## The problem

llm-compressor 0.10.x pins `transformers<=4.57.6`, but Qwen3.5 (`qwen3_5` model type) only exists in transformers 5.0+. Only **2 imports** actually break:

1. `TORCH_INIT_FUNCTIONS` removed from `transformers.modeling_utils` in 5.x
2. `Conv1D` moved from `transformers.modeling_utils` to `transformers.pytorch_utils`

## Patches

| File | Fix |
|---|---|
| `patches/dev.py.patch` | 3-tier import: `transformers.initialization` → `modeling_utils` → build from `torch.nn.init` |
| `patches/module.py.patch` | Try `pytorch_utils` first, fall back to `modeling_utils` |

## Apply

```bash
python apply_patches.py                          # auto-detect site-packages
python apply_patches.py /path/to/site-packages   # explicit path
python apply_patches.py --dry-run                 # preview only
```

Then install transformers 5.x:
```bash
pip install "transformers>=5.5" --no-deps
```

## MTP weight preservation

Qwen3.5's MTP (Multi-Token Prediction) weights are silently dropped during quantization because `Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`. The `inject_mtp_weights.py` script copies them back post-quantization:

```bash
python inject_mtp_weights.py --source /path/to/original --quantized /path/to/quantized
```

## Full example

```bash
python quantize_qwen35.py \
    --model huihui-ai/Huihui-Qwen3.5-27B-abliterated \
    --output ./Qwen3.5-27B-W4A16 \
    --scheme W4A16
```

## Verified compatibility

Tested against all 126 Python modules in llm-compressor 0.10.x with transformers 5.5.3. Zero import failures after patches.

## License

Apache-2.0 (same as llm-compressor)
