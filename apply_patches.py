#!/usr/bin/env python3
"""
Apply llm-compressor transformers 5.x compatibility patches.

Usage:
    python apply_patches.py /path/to/site-packages

This modifies llmcompressor in-place within the given site-packages directory.
"""

import argparse
import os
import re
import sys
import textwrap


def patch_dev_py(file_path: str) -> bool:
    """Patch utils/dev.py to handle TORCH_INIT_FUNCTIONS import across transformers versions."""
    with open(file_path, "r") as f:
        content = f.read()

    # The old pattern (existing llmcompressor code)
    old_block = textwrap.dedent("""\
        try:
            from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
        except ImportError:
            TORCH_INIT_FUNCTIONS = {}""")

    new_block = textwrap.dedent("""\
        try:
            # transformers >=5.x (if relocated)
            from transformers.initialization import TORCH_INIT_FUNCTIONS
        except ImportError:
            try:
                # transformers <5.0
                from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
            except ImportError:
                # Last resort: build the dict from torch.nn.init attributes.
                # An empty dict is NOT acceptable -- it makes skip_weights_initialize()
                # a no-op, silently corrupting quantization calibration.
                import torch
                TORCH_INIT_FUNCTIONS = {
                    name: getattr(torch.nn.init, name)
                    for name in dir(torch.nn.init)
                    if callable(getattr(torch.nn.init, name))
                    and name.endswith("_")
                    and not name.startswith("__")
                }""")

    if old_block in content:
        content = content.replace(old_block, new_block)
        with open(file_path, "w") as f:
            f.write(content)
        return True

    # Check if already patched
    if "transformers.initialization" in content:
        print(f"  [skip] {file_path} already patched")
        return True

    print(f"  [WARN] Could not find expected pattern in {file_path}", file=sys.stderr)
    return False


def patch_module_py(file_path: str) -> bool:
    """Patch utils/pytorch/module.py for Conv1D import compatibility."""
    with open(file_path, "r") as f:
        content = f.read()

    # Check if already using pytorch_utils (current versions already do this)
    if "from transformers.pytorch_utils import Conv1D" in content:
        print(f"  [skip] {file_path} already uses pytorch_utils import")
        return True

    # Old pattern: direct import from modeling_utils
    old_pattern = re.compile(
        r"try:\s*\n"
        r"\s+from transformers\.modeling_utils import Conv1D as TransformerConv1D\s*\n"
        r"\s*except.*?:\s*\n"
        r".*?TransformerConv1D\s*=\s*None",
        re.DOTALL,
    )

    new_block = textwrap.dedent("""\
        try:
            from transformers.pytorch_utils import Conv1D as TransformerConv1D
        except ImportError:
            try:
                from transformers.modeling_utils import Conv1D as TransformerConv1D
            except ImportError:
                TransformerConv1D = None""")

    match = old_pattern.search(content)
    if match:
        content = content[:match.start()] + new_block + content[match.end():]
        with open(file_path, "w") as f:
            f.write(content)
        return True

    print(f"  [WARN] Could not find expected pattern in {file_path}", file=sys.stderr)
    return False


def verify_imports(site_packages: str) -> bool:
    """Verify the patched imports work correctly."""
    print("\nVerifying imports...")
    import subprocess

    result = subprocess.run(
        [
            sys.executable, "-c",
            "from llmcompressor.utils.dev import TORCH_INIT_FUNCTIONS; "
            "assert len(TORCH_INIT_FUNCTIONS) > 0, "
            f"'TORCH_INIT_FUNCTIONS is empty! Got: {{}}';"
            "print(f'  TORCH_INIT_FUNCTIONS: {len(TORCH_INIT_FUNCTIONS)} entries OK')"
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": site_packages},
    )
    if result.returncode != 0:
        print(f"  [FAIL] TORCH_INIT_FUNCTIONS import: {result.stderr.strip()}", file=sys.stderr)
        return False
    print(result.stdout.strip())

    result = subprocess.run(
        [
            sys.executable, "-c",
            "from llmcompressor.utils.pytorch.module import TransformerConv1D; "
            "print(f'  TransformerConv1D: {TransformerConv1D}')"
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": site_packages},
    )
    if result.returncode != 0:
        print(f"  [FAIL] Conv1D import: {result.stderr.strip()}", file=sys.stderr)
        return False
    print(result.stdout.strip())

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply transformers 5.x compatibility patches to llm-compressor"
    )
    parser.add_argument(
        "site_packages",
        help="Path to the site-packages directory containing llmcompressor",
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Verify imports after patching (default: True)",
    )
    parser.add_argument(
        "--no-verify", action="store_false", dest="verify",
        help="Skip import verification",
    )
    args = parser.parse_args()

    sp = os.path.abspath(args.site_packages)
    llmc_root = os.path.join(sp, "llmcompressor")

    if not os.path.isdir(llmc_root):
        print(f"Error: {llmc_root} not found", file=sys.stderr)
        sys.exit(1)

    dev_py = os.path.join(llmc_root, "utils", "dev.py")
    module_py = os.path.join(llmc_root, "utils", "pytorch", "module.py")

    success = True
    print("Applying patches...")

    if os.path.exists(dev_py):
        print(f"  Patching {dev_py}")
        if not patch_dev_py(dev_py):
            success = False
    else:
        print(f"  [WARN] {dev_py} not found", file=sys.stderr)
        success = False

    if os.path.exists(module_py):
        print(f"  Patching {module_py}")
        if not patch_module_py(module_py):
            success = False
    else:
        print(f"  [WARN] {module_py} not found", file=sys.stderr)
        success = False

    if success and args.verify:
        if not verify_imports(sp):
            success = False

    if success:
        print("\nAll patches applied successfully.")
    else:
        print("\nPatching completed with warnings.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
