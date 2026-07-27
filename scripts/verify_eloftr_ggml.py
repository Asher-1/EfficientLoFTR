#!/usr/bin/env python3
"""RepVGG backbone parity for EfficientLoFTR GGUF (Phase 6b)."""

from __future__ import annotations

import argparse
import importlib.util
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def _load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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
    _load_module("loftr.utils.full_config", SRC_ROOT / "loftr/utils/full_config.py")


def load_torch_backbone(checkpoint: Path) -> torch.nn.Module:
    _ensure_loftr_modules()
    from loftr.backbone import build_backbone
    from loftr.backbone.repvgg import repvgg_model_convert
    from loftr.utils.full_config import full_default_cfg

    backbone = build_backbone(full_default_cfg)
    pl = types.ModuleType("pytorch_lightning")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    mc_mod = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class ModelCheckpoint:
        pass

    mc_mod.ModelCheckpoint = ModelCheckpoint
    callbacks.model_checkpoint = mc_mod
    pl.callbacks = callbacks
    for name, mod in [
        ("pytorch_lightning", pl),
        ("pytorch_lightning.callbacks", callbacks),
        ("pytorch_lightning.callbacks.model_checkpoint", mc_mod),
    ]:
        sys.modules[name] = mod

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    prefix = "matcher.backbone."
    backbone.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}, strict=True
    )
    backbone.eval()
    return repvgg_model_convert(backbone, save_path=None, do_copy=False)


def read_cpp_output(path: Path) -> tuple[np.ndarray, int, int, int]:
    data = path.read_bytes()
    if not data.startswith(b"ELOUT01\0"):
        raise ValueError(f"invalid ELOUT header: {path}")
    w, h, c = struct.unpack_from("<iii", data, 8)
    feat = np.frombuffer(data, dtype=np.float32, count=w * h * c, offset=20)
    return feat, w, h, c


def ggml_tensor_to_nchw(feat_flat: np.ndarray, w: int, h: int, c: int) -> np.ndarray:
    # ggml conv activations are stored linearly in CHW order (c major).
    return feat_flat.reshape(c, h, w)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ELoFTR RepVGG GGUF backbone")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "weights/eloftr_outdoor.ckpt")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--binary", default="cpp/build/eloftr_backbone")
    parser.add_argument("--output", type=Path, default=Path("/tmp/eloftr_cpp.elout"))
    parser.add_argument("--device", choices=["cpu", "cuda", "vulkan"], default="cpu")
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    if not args.gguf.is_file() or not args.checkpoint.is_file() or not args.image.is_file():
        print("missing input file", file=sys.stderr)
        return 2

    binary = REPO_ROOT / args.binary
    if not binary.is_file():
        print(f"C++ binary not found: {binary}", file=sys.stderr)
        return 77

    gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("failed to read image", file=sys.stderr)
        return 2
    if args.resize > 0:
        gray = cv2.resize(gray, (args.resize, args.resize), interpolation=cv2.INTER_LINEAR)
    h, w = gray.shape
    inp = (gray.astype(np.float32) / 255.0).reshape(-1)

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(inp.astype("<f4").tobytes())

    cmd = [
        str(binary),
        str(args.gguf),
        str(tmp_path),
        str(h),
        str(w),
        str(args.output),
        "--device",
        args.device,
    ]
    subprocess.run(cmd, check=True)
    cpp_feat_flat, fw, fh, fc = read_cpp_output(args.output)
    cpp_nchw = ggml_tensor_to_nchw(cpp_feat_flat, fw, fh, fc)[None, ...]

    model = load_torch_backbone(args.checkpoint)
    image_tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
    with torch.inference_mode():
        torch_out = model(image_tensor)["feats_c"]
    torch_np = torch_out.cpu().numpy()

    if torch_np.shape != cpp_nchw.shape:
        print(f"shape mismatch torch={torch_np.shape} cpp={cpp_nchw.shape}", file=sys.stderr)
        return 2

    abs_diff = np.abs(torch_np - cpp_nchw)
    print(
        f"feat_c abs diff: median={np.median(abs_diff):.6f} "
        f"p99={np.percentile(abs_diff, 99):.6f} max={np.max(abs_diff):.6f}"
    )
    corr = np.corrcoef(torch_np.reshape(-1), cpp_nchw.reshape(-1))[0, 1]
    print(f"corr={corr:.6f}")

    ok = np.median(abs_diff) <= args.tolerance and corr >= 0.9995
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
