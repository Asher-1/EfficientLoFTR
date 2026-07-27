# GGML validation — ELoFTR (2026-07-27)

Assets for **EfficientLoFTR RepVGG outdoor** (`eloftr_outdoor-*`).

| File | Description |
|------|-------------|
| `validation_matrix_eloftr.json` | F32/F16/Q8 × CPU/CUDA/Vulkan × 2 test images |
| `quantization_sizes.png` | F32 / F16 / Q8_0 GGUF sizes |
| `quantization_accuracy.png` | RepVGG corr vs PyTorch (CPU, Piazza) |
| `eloftr_parity_cpu.png` | Feature diff heatmap (PyTorch vs GGML, CPU) |

### Piazza reference (640²) — release parity

| Quant | CPU | CUDA | Vulkan |
|-------|-----|------|--------|
| F32 | PASS | PASS | PASS |
| F16 | PASS | PASS | PASS |
| Q8_0 | FAIL (experimental) | FAIL | FAIL |

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

Figures are embedded in the root [README.md](../README.md#ggml-validation-figures).
