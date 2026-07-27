#include "eloftr/eloftr_backbone.hpp"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace eloftr {
namespace {

using TensorMap = std::unordered_map<std::string, std::vector<float>>;

std::string Lower(std::string value) {
  for (char &c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

std::vector<float> ToGgmlConvWeight(const std::vector<float> &nchw, int32_t ic, int32_t oc,
                                    int32_t kh, int32_t kw) {
  std::vector<float> out(static_cast<size_t>(ic) * oc * kh * kw);
  for (int32_t o = 0; o < oc; ++o) {
    for (int32_t i = 0; i < ic; ++i) {
      for (int32_t y = 0; y < kh; ++y) {
        for (int32_t x = 0; x < kw; ++x) {
          const size_t pt_idx =
              static_cast<size_t>(o) * ic * kh * kw + static_cast<size_t>(i) * kh * kw +
                                  static_cast<size_t>(y) * kw + static_cast<size_t>(x);
          const size_t ggml_idx =
              static_cast<size_t>(x) + static_cast<size_t>(y) * kw +
              static_cast<size_t>(i) * kh * kw +
              static_cast<size_t>(o) * ic * kh * kw;
          out[ggml_idx] = nchw[pt_idx];
        }
      }
    }
  }
  return out;
}

struct PendingUpload {
  ggml_tensor *tensor = nullptr;
  std::vector<float> data;
};

ggml_tensor *MakeKernel(ggml_context *ctx, const std::vector<float> &nchw, int32_t kw,
                        int32_t kh, int32_t ic, int32_t oc,
                        std::vector<PendingUpload> *pending) {
  ggml_tensor *t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kw, kh, ic, oc);
  pending->push_back({t, ToGgmlConvWeight(nchw, ic, oc, kh, kw)});
  return t;
}

ggml_tensor *RepBlock(ggml_context *ctx, ggml_tensor *input, const TensorMap &weights,
                      const std::string &prefix, int32_t ic, int32_t oc, int32_t stride,
                      std::vector<PendingUpload> *pending, std::string *error) {
  const std::string wp = prefix + "_rbr_reparam_weight";
  const std::string bp = prefix + "_rbr_reparam_bias";
  if (weights.count(wp) == 0 || weights.count(bp) == 0) {
    if (error) {
      *error = "missing tensor " + wp;
    }
    return nullptr;
  }
  ggml_tensor *weight = MakeKernel(ctx, weights.at(wp), 3, 3, ic, oc, pending);
  ggml_tensor *bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, oc, 1);
  pending->push_back({bias, weights.at(bp)});
  ggml_tensor *y = ggml_conv_2d(ctx, weight, input, stride, stride, 1, 1, 1, 1);
  y = ggml_add(ctx, y, ggml_repeat(ctx, bias, y));
  return ggml_relu(ctx, y);
}

ggml_backend_t CreateBackend(const std::string &device, std::string *error) {
  ggml_backend_load_all();
  const std::string name = Lower(device);
  if (name.empty() || name == "cpu") {
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
  }
  if (name == "cuda" || name == "vulkan" || name == "gpu") {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
      ggml_backend_dev_t dev = ggml_backend_dev_get(i);
      if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        continue;
      }
      if (name != "gpu") {
        const char *registry = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (registry == nullptr || Lower(registry) != name) {
          continue;
        }
      }
      ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
      if (backend != nullptr) {
        return backend;
      }
    }
    if (error) {
      *error = "no usable backend: " + device;
    }
    return nullptr;
  }
  if (error) {
    *error = "unknown device: " + device;
  }
  return nullptr;
}

} // namespace

namespace {

constexpr uint32_t kDtypeF32 = 0;
constexpr uint32_t kDtypeF16 = 1;
constexpr uint32_t kDtypeQ8 = 2;

bool DequantPayload(uint32_t dtype, const std::vector<int64_t> &dims,
                    const std::vector<uint8_t> &payload, std::vector<float> *out) {
  int64_t count = 1;
  for (int64_t d : dims) {
    count *= d;
  }
  out->resize(static_cast<size_t>(count));
  if (dtype == kDtypeF32) {
    std::memcpy(out->data(), payload.data(), payload.size());
    return payload.size() == static_cast<size_t>(count) * sizeof(float);
  }
  if (dtype == kDtypeF16) {
    if (payload.size() != static_cast<size_t>(count) * sizeof(ggml_fp16_t)) {
      return false;
    }
    ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(payload.data()),
                          out->data(), count);
    return true;
  }
  if (dtype == kDtypeQ8) {
    const int64_t row = dims.empty() ? count : dims[0];
    const int64_t rows = count / row;
    const ggml_type_traits *traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    if (traits == nullptr || traits->to_float == nullptr) {
      return false;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, row);
    if (payload.size() != row_bytes * static_cast<size_t>(rows)) {
      return false;
    }
    for (int64_t r = 0; r < rows; ++r) {
      traits->to_float(payload.data() + r * row_bytes, out->data() + r * row, row);
    }
    return true;
  }
  return false;
}

} // namespace

bool LoadRepVggGguf(const std::string &path, RepVggTensorMap *tensors, std::string *error) {
  if (tensors == nullptr) {
    return false;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error) {
      *error = "failed to open GGUF";
    }
    return false;
  }
  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t count = 0;
  uint64_t meta = 0;
  in.read(reinterpret_cast<char *>(&magic), 4);
  in.read(reinterpret_cast<char *>(&version), 4);
  in.read(reinterpret_cast<char *>(&count), 8);
  in.read(reinterpret_cast<char *>(&meta), 8);
  if (magic != 0x46554747) {
    if (error) {
      *error = "invalid GGUF magic";
    }
    return false;
  }
  (void)version;
  for (uint64_t m = 0; m < meta; ++m) {
    uint64_t klen = 0;
    in.read(reinterpret_cast<char *>(&klen), 8);
    in.seekg(static_cast<std::streamoff>(klen), std::ios::cur);
    in.seekg(12, std::ios::cur);
  }
  tensors->clear();
  for (uint64_t t = 0; t < count; ++t) {
    uint64_t nlen = 0;
    in.read(reinterpret_cast<char *>(&nlen), 8);
    if (nlen > 4096) {
      if (error) {
        *error = "invalid tensor name length";
      }
      return false;
    }
    std::string name(static_cast<size_t>(nlen), '\0');
    in.read(name.data(), static_cast<std::streamoff>(nlen));
    uint32_t dtype = 0;
    uint32_t ndim = 0;
    in.read(reinterpret_cast<char *>(&dtype), 4);
    in.read(reinterpret_cast<char *>(&ndim), 4);
    std::vector<int64_t> dims(static_cast<size_t>(ndim));
    int64_t elems = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
      uint64_t dim = 0;
      in.read(reinterpret_cast<char *>(&dim), 8);
      dims[d] = static_cast<int64_t>(dim);
      elems *= dims[d];
    }
    size_t payload_bytes = 0;
    if (dtype == kDtypeF32) {
      payload_bytes = static_cast<size_t>(elems) * sizeof(float);
    } else if (dtype == kDtypeF16) {
      payload_bytes = static_cast<size_t>(elems) * sizeof(ggml_fp16_t);
    } else if (dtype == kDtypeQ8) {
      const int64_t row = dims.empty() ? elems : dims[0];
      payload_bytes = ggml_row_size(GGML_TYPE_Q8_0, row) *
                      static_cast<size_t>(elems / row);
    } else {
      if (error) {
        *error = "unsupported dtype";
      }
      return false;
    }
    std::vector<uint8_t> payload(payload_bytes);
    in.read(reinterpret_cast<char *>(payload.data()),
            static_cast<std::streamoff>(payload_bytes));
    std::vector<float> data;
    if (!DequantPayload(dtype, dims, payload, &data)) {
      if (error) {
        *error = "dequant failed: " + name;
      }
      return false;
    }
    (*tensors)[name] = std::move(data);
  }
  return true;
}

bool RunRepVggBackbone(const RepVggTensorMap &weights, const std::vector<float> &input_nchw,
                       int32_t h, int32_t w, const std::string &device,
                       std::vector<float> *feat_c_nchw, int32_t *out_h, int32_t *out_w,
                       int32_t *out_c, std::string *error) {
  if (feat_c_nchw == nullptr) {
    return false;
  }
  ggml_backend_t backend = CreateBackend(device, error);
  if (backend == nullptr) {
    return false;
  }

  TensorMap map(weights.begin(), weights.end());
  ggml_init_params params{128 * 1024 * 1024, nullptr, true};
  ggml_context *ctx = ggml_init(params);
  if (ctx == nullptr) {
    ggml_backend_free(backend);
    if (error) {
      *error = "ggml_init failed";
    }
    return false;
  }

  std::vector<PendingUpload> pending;
  ggml_tensor *inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, 1, 1);
  ggml_tensor *x = RepBlock(ctx, inp, map, "layer0", 1, 64, 2, &pending, error);
  if (x == nullptr) {
    ggml_free(ctx);
    ggml_backend_free(backend);
    return false;
  }
  if (const char *stop_after = std::getenv("ELOFTR_STOP_AFTER");
      stop_after != nullptr && std::string(stop_after) == "layer0") {
    goto run_graph;
  }
  x = RepBlock(ctx, x, map, "layer1_0", 64, 64, 1, &pending, error);
  x = RepBlock(ctx, x, map, "layer1_1", 64, 64, 1, &pending, error);
  x = RepBlock(ctx, x, map, "layer2_0", 64, 128, 2, &pending, error);
  for (int b = 1; b <= 3; ++b) {
    x = RepBlock(ctx, x, map, "layer2_" + std::to_string(b), 128, 128, 1, &pending, error);
  }
  x = RepBlock(ctx, x, map, "layer3_0", 128, 256, 2, &pending, error);
  for (int b = 1; b <= 13; ++b) {
    x = RepBlock(ctx, x, map, "layer3_" + std::to_string(b), 256, 256, 1, &pending, error);
  }
  if (x == nullptr) {
    ggml_free(ctx);
    ggml_backend_free(backend);
    return false;
  }

run_graph:
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, x);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buf == nullptr) {
    ggml_free(ctx);
    ggml_backend_free(backend);
    if (error) {
      *error = "backend alloc failed";
    }
    return false;
  }
  for (const PendingUpload &item : pending) {
    ggml_backend_tensor_set(item.tensor, item.data.data(), 0,
                            item.data.size() * sizeof(float));
  }
  ggml_backend_tensor_set(inp, input_nchw.data(), 0, input_nchw.size() * sizeof(float));
  if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    if (error) {
      *error = "graph compute failed";
    }
    return false;
  }

  const int64_t ow = x->ne[0];
  const int64_t oh = x->ne[1];
  const int64_t oc = x->ne[2];
  feat_c_nchw->resize(static_cast<size_t>(ow * oh * oc));
  ggml_backend_tensor_get(x, feat_c_nchw->data(), 0, feat_c_nchw->size() * sizeof(float));
  if (out_h) {
    *out_h = static_cast<int32_t>(oh);
  }
  if (out_w) {
    *out_w = static_cast<int32_t>(ow);
  }
  if (out_c) {
    *out_c = static_cast<int32_t>(oc);
  }
  ggml_backend_buffer_free(buf);
  ggml_backend_free(backend);
  ggml_free(ctx);
  return true;
}

} // namespace eloftr
