#!/usr/bin/env python3
"""End-to-end ELoFTR match benchmark: PyTorch vs GGML (CPU/CUDA/Vulkan).

Generates:
  - match_visualization.png   — stacked PyTorch + GGML match overlays
  - match_accuracy.png        — match count vs PyTorch reference
  - inference_latency.png     — avg inference time (ms)

Requires ACloudViewer build with test_eloftr_bench and libAICore.so.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


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
    for sub in ("loftr_module", "utils"):
        key = f"loftr.{sub}"
        if key not in sys.modules:
            pkg = types.ModuleType(key)
            pkg.__path__ = [str(SRC_ROOT / "loftr" / sub)]
            sys.modules[key] = pkg
    _load_module("loftr.utils.full_config", SRC_ROOT / "loftr/utils/full_config.py")
    _load_module("loftr.loftr", SRC_ROOT / "loftr/loftr.py")


def load_pytorch_matcher(checkpoint: Path, device: str):
    import torch

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.loftr.loftr import LoFTR, reparameter
    from src.loftr.utils.full_config import full_default_cfg

    cfg = deepcopy(full_default_cfg)
    matcher = LoFTR(config=cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    matcher.load_state_dict(ckpt["state_dict"])
    matcher = reparameter(matcher)
    matcher = matcher.eval()
    if device == "cuda" and torch.cuda.is_available():
        matcher = matcher.cuda()
    else:
        device = "cpu"
        matcher = matcher.cpu()
    return matcher, device


def resize_pair(img0: np.ndarray, img1: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    return cv2.resize(img0, (size, size)), cv2.resize(img1, (size, size))


def run_pytorch_match(matcher, device: str, img0: np.ndarray, img1: np.ndarray,
                      min_conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    import torch

    t0 = time.perf_counter()
    i0 = torch.from_numpy(img0)[None][None].float()
    i1 = torch.from_numpy(img1)[None][None].float()
    if device == "cuda":
        i0, i1 = i0.cuda(), i1.cuda()
    i0 /= 255.0
    i1 /= 255.0
    batch = {"image0": i0, "image1": i1}
    with torch.inference_mode():
        matcher.forward3(batch)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    mk0 = batch["mkpts0_f"].cpu().numpy()
    mk1 = batch["mkpts1_f"].cpu().numpy()
    conf = batch["mconf"].cpu().numpy()
    keep = conf >= min_conf
    return mk0[keep], mk1[keep], conf[keep], elapsed_ms


def parse_bench_stdout(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        if line.startswith("ELoFTR bench:"):
            for token in line.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    try:
                        out[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        out[k] = v
    return out


def run_ggml_bench(build_dir: Path, gguf: Path, device: str, img0: Path,
                   img1: Path, min_conf: float, matches_out: Path | None) -> dict:
    bench = build_dir / "bin" / "aicore_tests" / "test_eloftr_bench"
    if not bench.is_file():
        bench = build_dir / "bin" / "test_eloftr_bench"
    if not bench.is_file():
        raise FileNotFoundError(
            f"Missing {bench} — build ACloudViewer with AICore tests")

    env = os.environ.copy()
    lib_dirs = [str(build_dir / "bin"), str(build_dir / "bin" / "aicore_tests")]
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")])

    cmd = [
        str(bench),
        str(gguf),
        device,
        str(img0),
        str(img1),
        str(min_conf),
        "1",
        "3",
    ]
    if matches_out is not None:
        cmd += ["--matches-out", str(matches_out)]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    parsed = parse_bench_stdout(proc.stdout)
    parsed["exit_code"] = proc.returncode
    if proc.returncode != 0:
        parsed["stderr"] = proc.stderr.strip()
    return parsed


def load_ggml_matches(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    xs0, ys0, xs1, ys1, scores = [], [], [], [], []
    for m in data.get("matches", []):
        xs0.append(m["x0"])
        ys0.append(m["y0"])
        xs1.append(m["x1"])
        ys1.append(m["y1"])
        scores.append(m["score"])
    if not scores:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))
    return (
        np.stack([xs0, ys0], axis=1),
        np.stack([xs1, ys1], axis=1),
        np.asarray(scores, dtype=np.float32),
    )


def draw_matches(img0: np.ndarray, img1: np.ndarray, mk0: np.ndarray,
                 mk1: np.ndarray, conf: np.ndarray, title: str) -> np.ndarray:
    h, w = img0.shape
    canvas = np.zeros((h, w * 2), dtype=np.uint8)
    canvas[:, :w] = img0
    canvas[:, w:] = img1
    canvas_color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    n = min(len(mk0), 800)
    if n == 0:
        cv2.putText(canvas_color, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return canvas_color
    idx = np.argsort(-conf)[:n] if len(conf) else np.arange(n)
    for i in idx:
        p0 = (int(mk0[i, 0]), int(mk0[i, 1]))
        p1 = (int(mk1[i, 0] + w), int(mk1[i, 1]))
        cv2.circle(canvas_color, p0, 2, (255, 180, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas_color, p1, 2, (255, 180, 0), -1, cv2.LINE_AA)
        cv2.line(canvas_color, p0, p1, (46, 204, 113), 1, cv2.LINE_AA)
    cv2.putText(canvas_color, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas_color


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acloudviewer-build",
        type=Path,
        default=Path(os.environ.get("ACLOUDVIEWER_BUILD", "")),
        help="ACloudViewer build_app directory",
    )
    parser.add_argument("--gguf-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--quant", default="f16")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "weights/eloftr_outdoor.ckpt",
    )
    parser.add_argument(
        "--image0",
        type=Path,
        default=REPO_ROOT
        / "assets/phototourism_sample_images/piazza_san_marco_06795901_3725050516.jpg",
    )
    parser.add_argument(
        "--image1",
        type=Path,
        default=REPO_ROOT
        / "assets/phototourism_sample_images/piazza_san_marco_15148634_5228701572.jpg",
    )
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--min-conf", type=float, default=0.2)
    parser.add_argument("--devices", default="cpu,cuda,vulkan")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "assets/ggml_validation_20260727",
    )
    args = parser.parse_args()

    if not args.acloudviewer_build or not args.acloudviewer_build.is_dir():
        print("Set --acloudviewer-build or ACLOUDVIEWER_BUILD", file=sys.stderr)
        return 2
    if not args.image0.is_file() or not args.image1.is_file():
        print("image paths missing", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    img0 = cv2.imread(str(args.image0), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(args.image1), cv2.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        print("failed to read images", file=sys.stderr)
        return 2
    if args.resize > 0:
        img0, img1 = resize_pair(img0, img1, args.resize)

    gguf = args.gguf_dir / f"eloftr_outdoor-{args.quant}.gguf"
    if not gguf.is_file():
        print(f"missing GGUF: {gguf}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    pt_ok = args.checkpoint.is_file()
    conf_pt = None
    ref_matches = None
    pt_ms = float("nan")

    if pt_ok:
        try:
            matcher, pt_device = load_pytorch_matcher(args.checkpoint, "cuda")
            mk0_pt, mk1_pt, conf_pt, pt_ms = run_pytorch_match(
                matcher, pt_device, img0, img1, args.min_conf)
            ref_matches = int(len(conf_pt))
            rows.append({
                "backend": "pytorch",
                "device": pt_device,
                "quant": "fp32",
                "matches": ref_matches,
                "mean_conf": float(conf_pt.mean()) if ref_matches else 0.0,
                "avg_ms": round(pt_ms, 1),
            })
            vis_pt = draw_matches(
                img0, img1, mk0_pt, mk1_pt, conf_pt,
                f"PyTorch ({pt_device}) n={ref_matches} {pt_ms:.0f}ms")
            cv2.imwrite(str(args.output_dir / "match_visualization_pytorch.png"), vis_pt)
            pt_ok = True
        except Exception as exc:
            print(f"[WARN] PyTorch matcher failed: {exc} — GGML-only mode")
            pt_ok = False
    if not pt_ok:
        print(f"[WARN] PyTorch unavailable: {args.checkpoint}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        p0 = tmp_dir / "img0.png"
        p1 = tmp_dir / "img1.png"
        cv2.imwrite(str(p0), img0)
        cv2.imwrite(str(p1), img1)

        best_ggml_vis = None
        for device in [d.strip() for d in args.devices.split(",") if d.strip()]:
            matches_json = tmp_dir / f"matches_{device}.json"
            try:
                parsed = run_ggml_bench(
                    args.acloudviewer_build, gguf, device, p0, p1,
                    args.min_conf, matches_json)
            except FileNotFoundError as exc:
                print(exc, file=sys.stderr)
                return 77

            if parsed.get("exit_code", 1) != 0:
                rows.append({
                    "backend": "ggml",
                    "device": device,
                    "quant": args.quant,
                    "matches": 0,
                    "avg_ms": float("nan"),
                    "status": "FAIL",
                    "stderr": (parsed.get("stderr") or "")[-200:],
                })
                print(f"[FAIL] GGML {device} exit={parsed.get('exit_code')}")
                continue

            if not matches_json.is_file():
                rows.append({
                    "backend": "ggml",
                    "device": device,
                    "quant": args.quant,
                    "matches": int(parsed.get("matches", 0)),
                    "avg_ms": parsed.get("avg_ms", float("nan")),
                    "status": "OK",
                })
                continue

            mk0, mk1, conf = load_ggml_matches(matches_json)
            row = {
                "backend": "ggml",
                "device": device,
                "quant": args.quant,
                "matches": int(parsed.get("matches", len(conf))),
                "mean_conf": float(conf.mean()) if len(conf) else 0.0,
                "avg_ms": parsed.get("avg_ms", float("nan")),
                "status": "OK",
            }
            if ref_matches and ref_matches > 0:
                row["match_ratio_vs_pytorch"] = round(row["matches"] / ref_matches, 3)
            rows.append(row)
            print(f"[OK] GGML {device}: {row['matches']} matches {row['avg_ms']:.1f}ms")

            vis = draw_matches(
                img0, img1, mk0, mk1, conf,
                f"GGML {device.upper()} n={row['matches']} {row['avg_ms']:.0f}ms")
            cv2.imwrite(
                str(args.output_dir / f"match_visualization_ggml_{device}.png"), vis)
            if device == "cpu" or best_ggml_vis is None:
                best_ggml_vis = vis

    if best_ggml_vis is not None and pt_ok:
        pt_vis = cv2.imread(str(args.output_dir / "match_visualization_pytorch.png"))
        h, w, _ = pt_vis.shape
        combo = np.zeros((h * 2, w, 3), dtype=np.uint8)
        combo[:h] = pt_vis
        combo[h:] = best_ggml_vis
        cv2.imwrite(str(args.output_dir / "match_visualization.png"), combo)
    elif best_ggml_vis is not None:
        cv2.imwrite(str(args.output_dir / "match_visualization.png"), best_ggml_vis)

    report = args.output_dir / "match_benchmark_eloftr.json"
    report.write_text(json.dumps(rows, indent=2))

    ggml_rows = [r for r in rows if r["backend"] == "ggml" and r.get("status") == "OK"]
    if ggml_rows:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        labels = [r["device"] for r in ggml_rows]
        counts = [r["matches"] for r in ggml_rows]
        x = np.arange(len(labels))
        ax.bar(x, counts, color="#2a6f97", label="GGML")
        if ref_matches is not None:
            ax.axhline(ref_matches, color="#e76f51", linewidth=2,
                       label=f"PyTorch ({ref_matches})")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Match count")
        ax.set_title(
            f"End-to-end matches (min conf {args.min_conf})\n"
            "GGML = backbone coarse only (no transformer/fine yet)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "match_accuracy.png", dpi=160)
        plt.close(fig)

        lat_rows = [r for r in rows if np.isfinite(r.get("avg_ms", float("nan")))]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        labels = [f"{r['backend']}/{r['device']}" for r in lat_rows]
        times = [r["avg_ms"] for r in lat_rows]
        x = np.arange(len(labels))
        colors = ["#e76f51" if r["backend"] == "pytorch" else "#0b7a75" for r in lat_rows]
        ax.bar(x, times, color=colors)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel("Avg inference time (ms)")
        ax.set_title(f"ELoFTR latency ({args.quant}, {args.resize}px)")
        fig.tight_layout()
        fig.savefig(args.output_dir / "inference_latency.png", dpi=160)
        plt.close(fig)

    print("Wrote figures to", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
