#!/usr/bin/env python3
"""
inject_mtp_weights.py - Post-quantization MTP weight injector for Qwen3.5 models.

When llm-compressor quantizes Qwen3.5, the MTP (Multi-Token Prediction) weights are
silently dropped because the model class sets:

    Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected = [r"^mtp.*"]

This means from_pretrained() never loads them, and save_pretrained() never saves them.

This script merges the original FP16 MTP weights back into the quantized output.

Usage:
    python inject_mtp_weights.py \
        --source /path/to/original/Qwen3.5-27B \
        --quantized /path/to/quantized/output \
        [--dry-run]
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


MTP_PATTERN = re.compile(r"^mtp")


def find_safetensors_files(model_dir: str) -> list[str]:
    """Find all safetensors files in a model directory."""
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return files


def extract_mtp_tensors(source_dir: str) -> dict:
    """Extract all MTP tensors from source model safetensors files."""
    mtp_tensors = {}
    source_files = find_safetensors_files(source_dir)

    for sf_path in source_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if MTP_PATTERN.match(key):
                    mtp_tensors[key] = f.get_tensor(key)

    return mtp_tensors


def get_weight_map(quantized_dir: str) -> tuple[dict, str]:
    """Load the weight map from model.safetensors.index.json."""
    index_path = os.path.join(quantized_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return {}, index_path

    with open(index_path, "r") as f:
        index_data = json.load(f)

    return index_data, index_path


def inject_mtp_into_quantized(
    source_dir: str, quantized_dir: str, dry_run: bool = False
) -> int:
    """
    Inject MTP tensors from source into quantized model output.

    Strategy:
    - If quantized model uses a single model.safetensors file, append MTP tensors to it.
    - If quantized model is sharded (has model.safetensors.index.json), create a
      dedicated mtp_weights.safetensors shard and update the index.

    Returns the number of MTP tensors injected.
    """
    print(f"Source model: {source_dir}")
    print(f"Quantized model: {quantized_dir}")
    print()

    # Extract MTP tensors from source
    print("Extracting MTP tensors from source...")
    mtp_tensors = extract_mtp_tensors(source_dir)

    if not mtp_tensors:
        print("WARNING: No MTP tensors found in source model!", file=sys.stderr)
        return 0

    print(f"  Found {len(mtp_tensors)} MTP tensors")
    total_bytes = sum(t.numel() * t.element_size() for t in mtp_tensors.values())
    print(f"  Total size: {total_bytes / (1024**3):.2f} GB")
    print()

    if dry_run:
        print("[DRY RUN] Would inject the following tensors:")
        for name in sorted(mtp_tensors.keys())[:20]:
            t = mtp_tensors[name]
            print(f"  {name}: {list(t.shape)} ({t.dtype})")
        if len(mtp_tensors) > 20:
            print(f"  ... and {len(mtp_tensors) - 20} more")
        return len(mtp_tensors)

    # Determine quantized model layout
    index_data, index_path = get_weight_map(quantized_dir)
    single_file = os.path.join(quantized_dir, "model.safetensors")

    if isinstance(index_data, dict) and "weight_map" in index_data:
        # Sharded model - create a new shard for MTP weights
        mtp_shard_name = "mtp_weights.safetensors"
        mtp_shard_path = os.path.join(quantized_dir, mtp_shard_name)

        print(f"Writing MTP shard: {mtp_shard_path}")
        save_file(mtp_tensors, mtp_shard_path)

        # Update index
        weight_map = index_data["weight_map"]
        for tensor_name in mtp_tensors.keys():
            weight_map[tensor_name] = mtp_shard_name

        # Update metadata total_size if present
        if "metadata" in index_data and "total_size" in index_data["metadata"]:
            index_data["metadata"]["total_size"] = int(index_data["metadata"]["total_size"]) + total_bytes

        print(f"Updating index: {index_path}")
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
            f.write("\n")

    elif os.path.exists(single_file):
        # Single file model - load, merge, and rewrite
        print(f"Loading existing weights from {single_file}")
        existing_tensors = {}
        with safe_open(single_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                existing_tensors[key] = f.get_tensor(key)

        existing_tensors.update(mtp_tensors)

        print(f"Writing merged weights to {single_file}")
        # Preserve metadata
        save_file(existing_tensors, single_file)

    else:
        # No existing weights found - just create the MTP shard
        mtp_shard_name = "mtp_weights.safetensors"
        mtp_shard_path = os.path.join(quantized_dir, mtp_shard_name)
        print(f"Writing MTP weights to: {mtp_shard_path}")
        save_file(mtp_tensors, mtp_shard_path)

        # Create an index if one doesn't exist
        print(f"Creating index at: {index_path}")
        weight_map = {name: mtp_shard_name for name in mtp_tensors.keys()}
        index_data = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
            f.write("\n")

    # Update config.json to include mtp_num_hidden_layers if missing
    _update_config(source_dir, quantized_dir)

    return len(mtp_tensors)


def _update_config(source_dir: str, quantized_dir: str):
    """Copy mtp_num_hidden_layers from source config to quantized config if missing."""
    source_config_path = os.path.join(source_dir, "config.json")
    quant_config_path = os.path.join(quantized_dir, "config.json")

    if not os.path.exists(source_config_path) or not os.path.exists(quant_config_path):
        return

    with open(source_config_path, "r") as f:
        source_config = json.load(f)

    with open(quant_config_path, "r") as f:
        quant_config = json.load(f)

    updated = False

    # Check top-level config
    if "mtp_num_hidden_layers" in source_config and "mtp_num_hidden_layers" not in quant_config:
        quant_config["mtp_num_hidden_layers"] = source_config["mtp_num_hidden_layers"]
        updated = True
        print(f"  Added mtp_num_hidden_layers={source_config['mtp_num_hidden_layers']} to config.json")

    # Check text_config (nested config style)
    src_text_cfg = source_config.get("text_config", {})
    quant_text_cfg = quant_config.get("text_config", {})

    if "mtp_num_hidden_layers" in src_text_cfg:
        if "text_config" not in quant_config:
            quant_config["text_config"] = {}
        if "mtp_num_hidden_layers" not in quant_config["text_config"]:
            quant_config["text_config"]["mtp_num_hidden_layers"] = src_text_cfg["mtp_num_hidden_layers"]
            updated = True
            print(f"  Added text_config.mtp_num_hidden_layers={src_text_cfg['mtp_num_hidden_layers']} to config.json")

    if updated:
        with open(quant_config_path, "w") as f:
            json.dump(quant_config, f, indent=2)
            f.write("\n")


def verify_injection(quantized_dir: str) -> int:
    """Verify MTP tensors are present in the quantized output."""
    mtp_count = 0
    for sf_path in find_safetensors_files(quantized_dir):
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if MTP_PATTERN.match(key):
                    mtp_count += 1
    return mtp_count


def main():
    parser = argparse.ArgumentParser(
        description="Inject MTP weights from source model into quantized output"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the original FP16/BF16 model (with MTP weights)",
    )
    parser.add_argument(
        "--quantized",
        required=True,
        help="Path to the quantized model output (missing MTP weights)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"Error: source directory not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.quantized):
        print(f"Error: quantized directory not found: {args.quantized}", file=sys.stderr)
        sys.exit(1)

    injected = inject_mtp_into_quantized(args.source, args.quantized, dry_run=args.dry_run)

    if not args.dry_run and injected > 0:
        print()
        print("Verifying injection...")
        verified = verify_injection(args.quantized)
        print(f"  MTP tensors in quantized output: {verified}")
        if verified >= injected:
            print("  SUCCESS: All MTP tensors injected correctly.")
        else:
            print(f"  WARNING: Expected {injected}, found {verified}", file=sys.stderr)
            sys.exit(1)
    elif injected == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
