# EfficientLoFTR GGML integration (EfficientLoFTR repository)

Mirrors the DeepLSD / ALIKED per-repo policy: GGUF + GGML graph in this repo, optional ACloudViewer C API later.

## Layout

| Path | Role |
|------|------|
| `scripts/convert_eloftr_to_gguf.py` | PyTorch → GGUF (RepVGG + LoFTR weights) |
| `cpp/include/eloftr/eloftr.hpp` | Matcher API |
| `cpp/src/repvgg_ggml.cpp` | RepVGG backbone GGML graph (6b) |

## Milestones

| Step | Task | Gate |
|------|------|------|
| 6a | GGUF converter + API skeleton | tensors load |
| 6b | RepVGG + coarse/fine LoFTR GGML graph | match parity vs PyTorch |
| 6c | CUDA + Vulkan via ggml | ≥2× PyTorch, same matches |
| 6d | Optional `aicore_eloftr_*` in ACloudViewer | GUI smoke |

## Next (6b)

1. Fuse RepVGG conv+BN weights in converter (same as ALIKED `FuseConvBn`)
2. Build encoder pyramid + FPN-style features in `repvgg_ggml.cpp`
3. Port coarse transformer + fine matching to GGML (or hybrid CPU for attention v1)
4. `scripts/verify_eloftr_ggml.py` parity on MegaDepth / ScanNet pairs
