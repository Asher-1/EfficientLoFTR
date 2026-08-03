# GGML validation — ELoFTR (2026-07-27)

Assets for **EfficientLoFTR RepVGG outdoor** (`eloftr_outdoor-*`).

| File | Description |
|------|-------------|
| `validation_matrix_eloftr.json` | F32/F16/Q8 × CPU/CUDA/Vulkan × 2 test images |
| `quantization_sizes.png` | F32 / F16 / Q8_0 GGUF sizes |
| `quantization_accuracy.png` | RepVGG corr vs PyTorch (CPU, Piazza) |
| `eloftr_parity_cpu.png` | Feature diff heatmap (PyTorch vs GGML, CPU) |
| `match_visualization.png` | End-to-end match overlay (PyTorch vs GGML) |
| `match_accuracy.png` | Match count: GGML backends vs PyTorch |
| `inference_latency.png` | End-to-end inference time (ms) |
| `match_benchmark_eloftr.json` | Raw end-to-end benchmark rows |

### Piazza reference (640²) — RepVGG backbone parity

| Quant | CPU | CUDA | Vulkan |
|-------|-----|------|--------|
| F32 | PASS | PASS | PASS |
| F16 | PASS | PASS | PASS |
| Q8_0 | FAIL (experimental) | FAIL | FAIL |

### End-to-end match benchmark (phototourism pair, 640²)

Images: `piazza_san_marco_06795901_3725050516.jpg` vs `piazza_san_marco_15148634_5228701572.jpg` (`assets/phototourism_sample_images/`).

| Backend | Device | Matches | Avg ms | Notes |
|---------|--------|--------:|-------:|-------|
| PyTorch | CUDA | **726** | ~1670 | full LoFTR (coarse transformer + fine) |
| GGML F16 | CPU | 0 | ~17445 | backbone + CPU coarse only (Phase 6b) |
| GGML F16 | CUDA | 0 | ~19334 | same |
| GGML F16 | Vulkan | 0 | ~16811 | same |

GGML **0 matches is expected today**: ACloudViewer `aicore_eloftr_match_gray` runs RepVGG backbone + dual-softmax coarse on raw `feats_c` without `loftr_coarse` transformer or fine head. Raw backbone mutual coarse max conf ≪ 0.2 even on self-match (verified in PyTorch). See [cpp/BENCHMARK.md](../../cpp/BENCHMARK.md#visual-comparison).

Regenerate:

```bash
git submodule update --init third_party/ggml
cmake --build cpp/build --target eloftr_backbone -j4
export LD_LIBRARY_PATH=cpp/build/build-ggml/src:$LD_LIBRARY_PATH
python /path/to/ACloudViewer/core/AICore/scripts/run_ggml_validation_matrix.py \
  --model eloftr \
  --release-dir /path/to/ACloudViewer/core/AICore/models/release \
  --output-dir assets/ggml_validation_20260727
```

End-to-end match benchmark (requires ACloudViewer `test_eloftr_bench` + optional PyTorch ckpt):

```bash
export ACLOUDVIEWER_BUILD=/path/to/ACloudViewer/build_app
python scripts/generate_match_benchmark_figures.py \
  --image0 /path/to/image0.jpg \
  --image1 /path/to/image1.jpg \
  --checkpoint weights/eloftr_outdoor.ckpt \
  --output-dir assets/ggml_validation_20260727
```

Figures are embedded in the root [README.md](../README.md#ggml-validation-figures).
