#include <ATen/ATen.h>
#include <torch/types.h>
#include <typedef.h>

extern "C"
{
    ImageDims_t get_image_dims(const at::Tensor &tensor);

    ImageSize_t get_image_size(const at::Tensor &tensor);

    int64_t max(int64_t a, int64_t b);
    int64_t min(int64_t a, int64_t b);
    int64_t _max_value(torch::ScalarType dtype);

    torch::Dtype cvt_dtype2Dtype(torch_dtype dtype);
    torch_dtype cvt_Dtype2dtype(torch::Dtype dtype);
}
