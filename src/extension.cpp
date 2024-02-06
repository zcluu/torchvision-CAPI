#include <torch/extension.h>
#include <ATen/ATen.h>

#include "api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("resize", &api::resize, "Resize image");
    m.def("normalize", &api::normalize, "Normalize image");
    m.def("crop", &api::crop, "Crop image");
    m.def("center_crop", &api::center_crop, "Center crop image");
    m.def("resized_crop", &api::resized_crop, "Resized crop image");
}
