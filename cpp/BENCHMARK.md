# Validation snapshot

Date: 2026-07-27

Environment: AMD Ryzen 9 5950X, NVIDIA RTX 3060 12 GB, driver 550.x, ggml 0.17.0, CUDA 11.8 (arch 86), Vulkan via system `glslc`.

Test images: phototourism street scene (`example.png`) and Piazza San Marco (`piazza_san_marco_*.jpg`), 640×640 resize.

RepVGG backbone parity vs PyTorch `eloftr_outdoor.ckpt`:

| Quant | Backend | median abs | corr | Status |
|-------|---------|------------|------|--------|
| F32 | CPU | 0.00045 | 0.999988 | **release parity** |
| F32 | CUDA | 0.00133 | 0.999963 | **release parity** |
| F32 | Vulkan | 0.00183 | 0.999935 | **release parity** |
| F16 | CPU/CUDA/Vulkan | same as F32 | same | **release parity** |
| Q8_0 | CPU/CUDA/Vulkan | ~0.027 | ~0.974 | experimental (corr &lt; 0.9995) |

Tolerance: median ≤ 0.05, corr ≥ 0.9995 (`scripts/verify_eloftr_ggml.py`).

Full matrix: [`assets/ggml_validation_20260727/validation_matrix_eloftr.json`](../assets/ggml_validation_20260727/validation_matrix_eloftr.json).

## Model sizes (GGUF)

| Variant | Size | Notes |
|---------|-----:|-------|
| F32 | 32.7 MB | reference |
| F16 | 16.3 MB | ~2× smaller, parity with F32 |
| Q8_0 | 8.7 MB | ~3.8× smaller, experimental accuracy |

## Indoor model

**There is no official public EfficientLoFTR indoor checkpoint.**

| Checkpoint | Architecture | Released? |
|------------|--------------|-----------|
| `eloftr_outdoor.ckpt` | EfficientLoFTR RepVGG | ✅ [Google Drive](https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q?usp=sharing) |
| `indoor_ds*.ckpt` in LoFTR dataset links | **Original LoFTR** (not EfficientLoFTR) | ❌ incompatible state dict |
| ScanNet-trained EfficientLoFTR (paper indoor demo) | EfficientLoFTR | ❌ authors [do not plan to release](https://github.com/zju3dv/EfficientLoFTR/issues/35) |

Therefore **`eloftr_indoor-{f32,f16,q8_0}.gguf` are not shipped** in cloudViewer_downloads or qLightGlue. To add an indoor variant you must train with `scripts/reproduce_train/eloftr_outdoor.sh` on ScanNet (or your data), then:

```bash
python scripts/convert_eloftr_to_gguf.py \
  --checkpoint weights/your_indoor.ckpt \
  --output models/eloftr_indoor-f32.gguf
```

Outdoor weights may still work on some indoor pairs but with domain gap (see upstream README note).

## Visual comparison

See root [README.md](../README.md#ggml-validation-figures) for plots in `assets/ggml_validation_20260727/`.

End-to-end matcher (coarse RepVGG + CPU coarse match) is **Phase 6b** — full transformer/fine matching not yet in GGML.

## Build notes

Initialize ggml submodule first:

```bash
git submodule update --init third_party/ggml
```

CUDA on CUDA 11.8 (RTX 3060):

```bash
cmake -S cpp -B cpp/build -DELOFTR_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cpp/build --target eloftr_backbone -j4
```

Vulkan:

```bash
cmake -S cpp -B cpp/build -DELOFTR_GGML_VULKAN=ON -DVulkan_GLSLC_EXECUTABLE=/usr/local/bin/glslc
cmake --build cpp/build --target eloftr_backbone -j4
```

## Verify & release

```bash
python scripts/convert_eloftr_to_gguf.py \
  --checkpoint weights/eloftr_outdoor.ckpt \
  --output models/eloftr_outdoor-f32.gguf
aicore_gguf_quantize eloftr models/eloftr_outdoor-f32.gguf models/eloftr_outdoor-f16.gguf f16
aicore_gguf_quantize eloftr models/eloftr_outdoor-f32.gguf models/eloftr_outdoor-q8_0.gguf q8_0

python scripts/verify_eloftr_ggml.py \
  --gguf models/eloftr_outdoor-f32.gguf \
  --image assets/phototourism_sample_images/piazza_san_marco_06795901_3725050516.jpg \
  --device cpu|cuda|vulkan --resize 640
```

Regenerate validation assets:

```bash
python /path/to/ACloudViewer/core/AICore/scripts/run_ggml_validation_matrix.py \
  --model eloftr \
  --output-dir assets/ggml_validation_20260727
```

GGUF release: [cloudViewer_downloads / ELoFTR](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/ELoFTR)
