#include <ATen/ATen.h>
#include <typedef.h>

ImageDims_t get_image_dims(const at::Tensor& tensor);

ImageSize_t get_image_size(const at::Tensor& tensor);

int64_t max(int64_t a, int64_t b);
int64_t min(int64_t a, int64_t b);