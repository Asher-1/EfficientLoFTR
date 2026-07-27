#include "eloftr/eloftr_backbone.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool ReadRawFloats(const std::string &path, std::vector<float> *data) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff bytes = in.tellg();
  in.seekg(0, std::ios::beg);
  if (bytes <= 0 || bytes % 4 != 0) {
    return false;
  }
  data->resize(static_cast<size_t>(bytes / 4));
  in.read(reinterpret_cast<char *>(data->data()), bytes);
  return true;
}

bool WriteOutput(const std::string &path, int32_t w, int32_t h, int32_t c,
                 const std::vector<float> &feat) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  const char header[] = "ELOUT01\0";
  out.write(header, 8);
  out.write(reinterpret_cast<const char *>(&w), 4);
  out.write(reinterpret_cast<const char *>(&h), 4);
  out.write(reinterpret_cast<const char *>(&c), 4);
  out.write(reinterpret_cast<const char *>(feat.data()),
            static_cast<std::streamoff>(feat.size() * sizeof(float)));
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 7) {
    std::cerr << "usage: " << argv[0]
              << " <model.gguf> <input.raw> <H> <W> <output.elout> [--device cpu|cuda|vulkan]\n";
    return 1;
  }
  const int32_t h = std::stoi(argv[3]);
  const int32_t w = std::stoi(argv[4]);
  std::string device = "cpu";
  for (int i = 6; i < argc - 1; ++i) {
    if (std::string(argv[i]) == "--device") {
      device = argv[i + 1];
    }
  }

  eloftr::RepVggTensorMap weights;
  std::string error;
  if (!eloftr::LoadRepVggGguf(argv[1], &weights, &error)) {
    std::cerr << error << "\n";
    return 1;
  }

  std::vector<float> input;
  if (!ReadRawFloats(argv[2], &input) ||
      static_cast<int32_t>(input.size()) != h * w) {
    std::cerr << "invalid input.raw size\n";
    return 1;
  }

  std::vector<float> feat;
  int32_t oh = 0;
  int32_t ow = 0;
  int32_t oc = 0;
  if (!eloftr::RunRepVggBackbone(weights, input, h, w, device, &feat, &oh, &ow, &oc, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  if (!WriteOutput(argv[5], ow, oh, oc, feat)) {
    std::cerr << "failed to write output\n";
    return 1;
  }
  std::cout << "{\"feat_h\":" << oh << ",\"feat_w\":" << ow << ",\"feat_c\":" << oc
            << ",\"device\":\"" << device << "\"}\n";
  return 0;
}
