#include <utils.h>

ImageDims_t get_image_dims(const at::Tensor &tensor)
{
    ImageDims_t dims;
    dims.channel = tensor.size(0);
    dims.width = tensor.size(2);
    dims.height = tensor.size(1);
    return dims;
}

ImageSize_t get_image_size(const at::Tensor &tensor)
{
    ImageSize_t size;
    size.first = tensor.size(1);
    size.second = tensor.size(2);
    tensor.options();
    return size;
}

int64_t max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

int64_t min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

int64_t _max_value(torch::ScalarType dtype)
{
    if (dtype == torch::kUInt8)
    {
        return 255;
    }
    if (dtype == torch::kInt8)
    {
        return 127;
    }
    if (dtype == torch::kInt16)
    {
        return 32767;
    }
    if (dtype == torch::kInt32)
    {
        return 2147483647;
    }
    if (dtype == torch::kInt64)
    {
        return 9223372036854775807;
    }
    return 1;
}