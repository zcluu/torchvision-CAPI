#include <ATen/ATen.h>
#include <api.h>

// extern "C"
// {
//   namespace api
//   {
//     at::Tensor resize(const at::Tensor &src, ImageSize_t size);
//     at::Tensor normalize(at::Tensor &tensor, std::vector<float> &mean, std::vector<float> &std, bool inplace);
//     at::Tensor crop(at::Tensor &img, int64_t top, int64_t left, int64_t height, int64_t width);
//     at::Tensor center_crop(at::Tensor &img, ImageSize_t output_size);
//     at::Tensor resized_crop(
//         at::Tensor &img,
//         int64_t top,
//         int64_t left,
//         int64_t height,
//         int64_t width,
//         ImageSize_t resize);
//     at::Tensor hflip(at::Tensor &img);
//     at::Tensor vflip(at::Tensor &img);
//     at::Tensor rgb_to_grayscale(at::Tensor &img, int num_output_channels);

//     // Adjust Functions
//     at::Tensor adjust_brightness(at::Tensor &img, float brightness_factor);
//     at::Tensor adjust_contrast(at::Tensor &img, float contrast_factor);
//     at::Tensor adjust_saturation(at::Tensor &img, float saturation_factor);

//     // Other Functions
//     at::Tensor convert_image_dtype(at::Tensor &img, torch_dtype _dtype);
//   }
// }

namespace api
{
  void resize(at::Tensor &src, at::Tensor &dst)
  {
    ImageSize_t size = get_image_size(dst);
    dst = resize_func(src, size);
  }

  void normalize(at::Tensor &tensor, float *_mean, float *_std, bool inplace, at::Tensor &dst)
  {
    std::vector<float> mean = {_mean[0], _mean[1], _mean[2]};
    std::vector<float> std = {_std[0], _std[1], _std[2]};
    dst = normalize_func(tensor, mean, std, inplace);
  }

  void crop(at::Tensor &src, int64_t top, int64_t left, int64_t height, int64_t width, at::Tensor &dst)
  {
    dst = crop_func(src, top, left, height, width);
  }

  void center_crop(at::Tensor &img, at::Tensor &dst)
  {
    ImageSize_t output_size = get_image_size(dst);
    dst = center_crop_func(img, output_size);
  }

  at::Tensor resized_crop(
      at::Tensor &img,
      int64_t top,
      int64_t left,
      int64_t height,
      int64_t width,
      at::Tensor &dst)
  {
    ImageSize_t resize = get_image_size(dst);
    return resized_crop_func(img, top, left, height, width, resize);
  }

  void hflip(at::Tensor &img, at::Tensor &dst)
  {
    dst = hflip_func(img);
    return;
  }

  void vflip(at::Tensor &img, at::Tensor &dst)
  {
    dst = vflip_func(img);
  }
  at::Tensor rgb_to_grayscale(at::Tensor &img, int num_output_channels)
  {
    return rgb_to_grayscale_func(img, num_output_channels);
  }
  at::Tensor adjust_brightness(at::Tensor &img, float brightness_factor)
  {
    return adjust_brightness_func(img, brightness_factor);
  }
  at::Tensor adjust_contrast(at::Tensor &img, float contrast_factor)
  {
    return adjust_contrast_func(img, contrast_factor);
  }
  at::Tensor adjust_saturation(at::Tensor &img, float saturation_factor)
  {
    return adjust_saturation_func(img, saturation_factor);
  }
  at::Tensor convert_image_dtype(at::Tensor &img, torch_dtype _dtype)
  {
    return convert_image_dtype_func(img, _dtype);
  }
}
