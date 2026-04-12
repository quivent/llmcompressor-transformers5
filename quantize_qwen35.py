#!/usr/bin/env python3
"""
quantize_qwen35.py - Complete example: quantize Qwen3.5-27B with MTP preservation.

This script demonstrates the full workflow:
1. Apply transformers 5.x compatibility patches (if needed)
2. Quantize the model using llm-compressor with FP8 (W8A8)
3. Inject MTP weights back into the quantized output

Prerequisites:
    pip install llmcompressor transformers>=4.56.1 torch safetensors

Usage:
    python quantize_qwen35.py \
        --model Qwen/Qwen3.5-27B \
        --output ./Qwen3.5-27B-FP8 \
        [--scheme FP8] \
        [--num-calibration-samples 512] \
        [--max-seq-len 4096]
"""

import argparse
import os
import sys
from pathlib import Path


def apply_patches_if_needed():
    """Apply transformers 5.x compat patches if TORCH_INIT_FUNCTIONS is broken."""
    try:
        from llmcompressor.utils.dev import TORCH_INIT_FUNCTIONS

        if len(TORCH_INIT_FUNCTIONS) == 0:
            raise ImportError("TORCH_INIT_FUNCTIONS is empty")
        print(f"[OK] TORCH_INIT_FUNCTIONS has {len(TORCH_INIT_FUNCTIONS)} entries")
    except ImportError as e:
        print(f"[PATCH] Need to apply patches: {e}")
        import site

        sp = site.getsitepackages()[0]
        patch_script = os.path.join(os.path.dirname(__file__), "apply_patches.py")
        ret = os.system(f"{sys.executable} {patch_script} {sp}")
        if ret != 0:
            print("ERROR: Failed to apply patches", file=sys.stderr)
            sys.exit(1)


def quantize_model(
    model_path: str,
    output_path: str,
    scheme: str = "FP8",
    num_samples: int = 512,
    max_seq_len: int = 4096,
):
    """Run the quantization pipeline."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"\n{'='*60}")
    print(f"Quantizing: {model_path}")
    print(f"Output:     {output_path}")
    print(f"Scheme:     {scheme}")
    print(f"Samples:    {num_samples}")
    print(f"Max seq:    {max_seq_len}")
    print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare calibration dataset
    print("Preparing calibration data...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(
            tokenizer(
                example["text"],
                truncation=True,
                max_length=max_seq_len,
                padding=False,
                return_tensors="pt",
            )
        )

    # Define quantization recipe
    if scheme == "FP8":
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="FP8",
            ignore=["lm_head"],
        )
    elif scheme == "W8A8":
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=["lm_head"],
        )
    elif scheme == "W4A16":
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
        )
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Use FP8, W8A8, or W4A16.")

    # Run quantization
    print("Running quantization...")
    oneshot(
        model=model,
        dataset=samples,
        recipe=recipe,
        output_dir=output_path,
        max_seq_length=max_seq_len,
    )

    print(f"\nQuantization complete. Output at: {output_path}")
    return output_path


def inject_mtp(source_path: str, quantized_path: str):
    """Inject MTP weights from source into quantized output."""
    inject_script = os.path.join(os.path.dirname(__file__), "inject_mtp_weights.py")
    print(f"\n{'='*60}")
    print("Injecting MTP weights...")
    print(f"{'='*60}\n")
    ret = os.system(
        f"{sys.executable} {inject_script} "
        f"--source {source_path} --quantized {quantized_path}"
    )
    if ret != 0:
        print("ERROR: MTP injection failed", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3.5 with MTP weight preservation"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-27B",
        help="Model path or HuggingFace hub ID (default: Qwen/Qwen3.5-27B)",
    )
    parser.add_argument(
        "--output",
        default="./Qwen3.5-27B-FP8",
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--scheme",
        default="FP8",
        choices=["FP8", "W8A8", "W4A16"],
        help="Quantization scheme (default: FP8)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length for calibration (default: 4096)",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip quantization, only inject MTP weights (useful for retry)",
    )
    parser.add_argument(
        "--skip-mtp-inject",
        action="store_true",
        help="Skip MTP injection (model does not use MTP)",
    )
    args = parser.parse_args()

    # Step 1: Ensure patches are applied
    apply_patches_if_needed()

    # Step 2: Quantize
    if not args.skip_quantize:
        quantize_model(
            model_path=args.model,
            output_path=args.output,
            scheme=args.scheme,
            num_samples=args.num_calibration_samples,
            max_seq_len=args.max_seq_len,
        )
    else:
        print(f"[SKIP] Quantization skipped, using existing output: {args.output}")

    # Step 3: Inject MTP weights
    if not args.skip_mtp_inject:
        # Resolve source path (could be a hub ID or local path)
        source_path = args.model
        if not os.path.isdir(source_path):
            from huggingface_hub import snapshot_download

            print(f"Downloading source model for MTP extraction: {source_path}")
            source_path = snapshot_download(repo_id=args.model)

        inject_mtp(source_path, args.output)
    else:
        print("[SKIP] MTP injection skipped")

    print(f"\nDone! Quantized model with MTP weights: {args.output}")
    print(f"\nTo use with vLLM:")
    print(f"  vllm serve {args.output} --num-speculative-tokens 1 --trust-remote-code")


if __name__ == "__main__":
    main()
