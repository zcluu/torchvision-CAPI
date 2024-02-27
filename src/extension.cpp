#include <torch/extension.h>
#include <ATen/ATen.h>

#include "api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::enum_<torch_dtype>(m, "DataType")
        .value("float32", torch_dtype::float32)
        .value("float64", torch_dtype::float64)
        .value("int32", torch_dtype::int32)
        .value("int64", torch_dtype::int64)
        .value("uint8", torch_dtype::uint8)
        .value("int8", torch_dtype::int8)
        .value("int16", torch_dtype::int16)
        .value("_float", torch_dtype::_float)
        .value("_int", torch_dtype::_int);
    m.def("resize", &api::resize, "Resize image");
    m.def("normalize", &api::normalize, "Normalize image");
    m.def("crop", &api::crop, "Crop image");
    m.def("center_crop", &api::center_crop, "Center crop image");
    m.def("resized_crop", &api::resized_crop, "Resized crop image");
    m.def("hflip", &api::hflip, "Horizontally flip image");
    m.def("vflip", &api::vflip, "Vertically flip image");
    m.def("rgb_to_grayscale", &api::rgb_to_grayscale, "Convert RGB image to grayscale");

    m.def("adjust_brightness", &api::adjust_brightness, "Adjust brightness of an image");
    m.def("adjust_contrast", &api::adjust_contrast, "Adjust contrast of an image");
    m.def("adjust_saturation", &api::adjust_saturation, "Adjust saturation of an image");

    m.def("convert_image_dtype", &api::convert_image_dtype, "Convert image dtype");
}
