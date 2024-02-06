#include <ATen/ATen.h>
#include <torch/nn/functional/upsampling.h>
#include <torch/nn/functional/padding.h>
#include <typedef.h>
#include <utils.h>

#include <iostream>

void resize_func(const at::Tensor& src, at::Tensor& dst, ImageSize_t resize);
at::Tensor normalize_func(at::Tensor& tensor, std::vector<float>& mean, std::vector<float>& std, bool inplace);
at::Tensor crop_func(at::Tensor& img, int64_t top, int64_t left, int64_t height, int64_t width);
at::Tensor center_crop_func(at::Tensor& img, ImageSize_t output_size);