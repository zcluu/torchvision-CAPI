#include <torch/extension.h>
#include <ATen/ATen.h>
#include <api.h>

namespace api {
  at::Tensor resize(const at::Tensor& src, ImageSize_t new_size) {
    at::Tensor dst = torch::empty({ new_size.first, new_size.second }, src.options());
    resize_func(src, dst, new_size);
    return dst;
  }
  at::Tensor normalize(at::Tensor& tensor, std::vector<float>& mean, std::vector<float>& std, bool inplace) {
    return normalize_func(tensor, mean, std, inplace);
  }
  at::Tensor crop(at::Tensor& img, int64_t top, int64_t left, int64_t height, int64_t width) {
    return crop_func(img, top, left, height, width);
  }
  at::Tensor center_crop(at::Tensor& img, ImageSize_t output_size) {
    return center_crop_func(img, output_size);
  }
}
