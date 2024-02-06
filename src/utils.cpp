#include <utils.h>

ImageDims_t get_image_dims(const at::Tensor& tensor) {
    ImageDims_t dims;
    dims.channel = tensor.size(0);
    dims.width = tensor.size(2);
    dims.height = tensor.size(1);
    return dims;
}

ImageSize_t get_image_size(const at::Tensor& tensor) {
    ImageSize_t size;
    size.first = tensor.size(1);
    size.second = tensor.size(2);
    return size;
}

int64_t max(int64_t a, int64_t b) {
    return a > b ? a : b;
}

int64_t min(int64_t a, int64_t b) {
    return a < b ? a : b;
}