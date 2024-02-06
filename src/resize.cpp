#include <functional.h>

namespace F = torch::nn::functional;

at::Tensor resize_func(const at::Tensor &src, ImageSize_t resize)
{
    assert(src.dim() == 3);
    long H = src.size(1);
    long W = src.size(2);

    float scale_h = resize.first / static_cast<float>(H);
    float scale_w = resize.second / static_cast<float>(W);

    double scale = std::min({scale_h, scale_w});
    ImageSize_t scale_to{static_cast<long>(H * scale), static_cast<long>(W * scale)};
    // resize image
    at::Tensor resized_img = F::interpolate(
                                 src.unsqueeze(0),
                                 F::InterpolateFuncOptions()
                                     .size(std::vector<int64_t>({scale_to.first, scale_to.second}))
                                     .mode(torch::kBilinear)
                                     .antialias(true)
                                     .align_corners(false))
                                 .squeeze(0)
                                 .clamp_(0, 1);
    resized_img = F::pad(resized_img, F::PadFuncOptions({0, resize.second - scale_to.second, 0, resize.first - scale_to.first}));
    return resized_img;
}
