#!/usr/bin/env python3
"""Inspect an RSL-RL/Isaac Lab PyTorch checkpoint (model_XXX.pt).

Usage:
  python scripts/inspect_checkpoint.py --ckpt /absolute/path/to/model_50.pt

Prints top-level keys and shapes/dtypes of tensors inside.
"""
import argparse
import os
from typing import Any

import torch


def summarize_item(k: str, v: Any) -> str:
    try:
        import numpy as np  # noqa: F401
    except Exception:
        pass
    if torch.is_tensor(v):
        return f"Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
    if isinstance(v, (int, float, str, bool)):
        return f"{type(v).__name__} value={v}"
    if isinstance(v, dict):
        return f"dict({len(v)})"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}({len(v)})"
    return type(v).__name__


def main():
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint file (model_XXX.pt)")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file, e.g., .../model_50.pt")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")

    if isinstance(obj, dict):
        print("[INFO] Top-level keys:")
        for k in obj.keys():
            print(f"  - {k}: {summarize_item(k, obj[k])}")
        # Common containers
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            print("\n[INFO] Model state_dict parameters:")
            for i, (name, tensor) in enumerate(obj["model_state_dict"].items()):
                if isinstance(tensor, torch.Tensor):
                    print(f"  [{i:04d}] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
                else:
                    print(f"  [{i:04d}] {name}: {type(tensor).__name__}")
        if "policy" in obj and hasattr(obj["policy"], "actor_critic"):
            # rsl-rl sometimes stores policy object; we avoid printing huge internals
            print("\n[INFO] Found 'policy' object (actor_critic present). Not expanding to avoid spam.")
    else:
        print(f"[WARN] Unexpected checkpoint type: {type(obj).__name__}")

    print("\n[HINT] Use your play script to visualize with --checkpoint <path>.")


if __name__ == "__main__":
    main()
