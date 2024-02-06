#include <ATen/ATen.h>
#include <typedef.h>
#include <functional.h>

namespace api
{
    at::Tensor resize(const at::Tensor &src, ImageSize_t new_size);
    at::Tensor normalize(at::Tensor &tensor, std::vector<float> &mean, std::vector<float> &std, bool inplace = false);
    at::Tensor crop(at::Tensor &img, int64_t top, int64_t left, int64_t height, int64_t width);
    at::Tensor center_crop(at::Tensor &img, ImageSize_t output_size);
    at::Tensor resized_crop(
        at::Tensor &img,
        int64_t top,
        int64_t left,
        int64_t height,
        int64_t width,
        ImageSize_t resize);
}
