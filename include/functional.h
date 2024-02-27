#include <ATen/ATen.h>
#include <torch/nn/functional/upsampling.h>
#include <torch/nn/functional/padding.h>
#include <typedef.h>
#include <utils.h>

#include <iostream>

extern "C"
{
    at::Tensor resize_func(at::Tensor &src, ImageSize_t resize);
    at::Tensor normalize_func(at::Tensor &tensor, std::vector<float> &mean, std::vector<float> &std, bool inplace);
    // Crop Functions
    at::Tensor crop_func(at::Tensor &img, int64_t top, int64_t left, int64_t height, int64_t width);
    at::Tensor center_crop_func(at::Tensor &img, ImageSize_t output_size);
    at::Tensor resized_crop_func(
        at::Tensor &img,
        int64_t top,
        int64_t left,
        int64_t height,
        int64_t width,
        ImageSize_t resize);
    // Flip Functions
    at::Tensor hflip_func(at::Tensor &img);
    at::Tensor vflip_func(at::Tensor &img);

    // Adjust Funcstions
    at::Tensor adjust_brightness_func(at::Tensor &img, float brightness_factor);
    at::Tensor adjust_contrast_func(at::Tensor &img, float contrast_factor);
    at::Tensor adjust_saturation_func(at::Tensor &img, float saturation_factor);
    // at::Tensor adjust_hue_func(at::Tensor &img, float hue_factor);

    at::Tensor blend(at::Tensor &img1, at::Tensor &img2, float ratio);

    // CVT Color
    at::Tensor rgb_to_grayscale_func(at::Tensor &img, int num_output_channels = 1);

    // Other Functions
    at::Tensor convert_image_dtype_func(at::Tensor &img, torch_dtype _dtype);
}