#include <ATen/ATen.h>
#include <typedef.h>
#include <functional.h>

extern "C"
{
    namespace api
    {
        void resize(at::Tensor &src, at::Tensor &dst);
        void normalize(at::Tensor &src, float *mean, float *std, bool inplace, at::Tensor &dst);
        void crop(at::Tensor &img, int64_t top, int64_t left, int64_t height, int64_t width, at::Tensor &dst);
        void center_crop(at::Tensor &img, at::Tensor &dst);
        at::Tensor resized_crop(
            at::Tensor &img,
            int64_t top,
            int64_t left,
            int64_t height,
            int64_t width,
            at::Tensor &dst);
        void hflip(at::Tensor &img, at::Tensor &dst);
        void vflip(at::Tensor &img, at::Tensor &dst);
        at::Tensor rgb_to_grayscale(at::Tensor &img, int num_output_channels = 1);

        // Adjust Functions
        at::Tensor adjust_brightness(at::Tensor &img, float brightness_factor);
        at::Tensor adjust_contrast(at::Tensor &img, float contrast_factor);
        at::Tensor adjust_saturation(at::Tensor &img, float saturation_factor);

        // Other Functions
        at::Tensor convert_image_dtype(at::Tensor &img, torch_dtype _dtype);
    }
}
