#!/usr/bin/env python3
"""Convert EfficientLoFTR checkpoint to GGUF with fused RepVGG blocks."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import struct
import sys
import types
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0


def _write_string(f, s: str) -> None:
    data = s.encode("utf-8")
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


def _write_tensor(f, name: str, arr: np.ndarray) -> None:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    _write_string(f, name)
    f.write(struct.pack("<I", GGML_TYPE_F32))
    f.write(struct.pack("<I", len(arr.shape)))
    for dim in arr.shape:
        f.write(struct.pack("<Q", int(dim)))
    f.write(flat.tobytes())


def _load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_loftr_modules() -> None:
    if "loftr" not in sys.modules:
        pkg = types.ModuleType("loftr")
        pkg.__path__ = [str(SRC_ROOT / "loftr")]
        sys.modules["loftr"] = pkg
    if "loftr.backbone" not in sys.modules:
        pkg = types.ModuleType("loftr.backbone")
        pkg.__path__ = [str(SRC_ROOT / "loftr" / "backbone")]
        sys.modules["loftr.backbone"] = pkg
    _load_module("loftr.backbone.repvgg", SRC_ROOT / "loftr/backbone/repvgg.py")
    _load_module("loftr.backbone.backbone", SRC_ROOT / "loftr/backbone/backbone.py")
    _load_module("loftr.backbone", SRC_ROOT / "loftr/backbone/__init__.py")
    if "loftr.utils" not in sys.modules:
        pkg = types.ModuleType("loftr.utils")
        pkg.__path__ = [str(SRC_ROOT / "loftr" / "utils")]
        sys.modules["loftr.utils"] = pkg
    _load_module("loftr.utils.full_config", SRC_ROOT / "loftr/utils/full_config.py")


def fuse_repvgg(model: torch.nn.Module) -> torch.nn.Module:
    from loftr.backbone.repvgg import repvgg_model_convert

    return repvgg_model_convert(copy.deepcopy(model), save_path=None, do_copy=False)


def collect_tensors(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in model.state_dict().items():
        if not torch.is_floating_point(value):
            continue
        out[key.replace(".", "_")] = value.detach().cpu().numpy().astype(np.float32)
    return out


def _ensure_pytorch_lightning_stub() -> None:
    if "pytorch_lightning" in sys.modules:
        return
    pl = types.ModuleType("pytorch_lightning")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    model_checkpoint = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class ModelCheckpoint:  # noqa: N801 - matches checkpoint class name
        pass

    model_checkpoint.ModelCheckpoint = ModelCheckpoint
    callbacks.model_checkpoint = model_checkpoint
    pl.callbacks = callbacks
    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = model_checkpoint


def load_model(checkpoint: Path, device: str = "cpu") -> torch.nn.Module:
    _ensure_pytorch_lightning_stub()
    _ensure_loftr_modules()
    from loftr.backbone import build_backbone
    from loftr.utils.full_config import full_default_cfg

    backbone = build_backbone(full_default_cfg)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    prefix = "matcher.backbone."
    backbone_state = {
        key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
    }
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()
    return backbone


def convert(checkpoint: Path, output: Path) -> None:
    backbone = load_model(checkpoint)
    fused = fuse_repvgg(backbone)
    tensors = collect_tensors(fused)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", 0))
        for name, tensor in sorted(tensors.items()):
            _write_tensor(f, name, tensor)
    print(f"Wrote {len(tensors)} tensors (RepVGG deploy-fused) -> {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert EfficientLoFTR to GGUF")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
