#include <functional.h>

torch::Dtype cvt_dtype2Dtype(torch_dtype dtype)
{
    switch (dtype)
    {
    case torch_dtype::float32:
        return torch::kFloat32;
    case torch_dtype::float64:
        return torch::kFloat64;
    case torch_dtype::int32:
        return torch::kInt32;
    case torch_dtype::int64:
        return torch::kInt64;
    case torch_dtype::uint8:
        return torch::kUInt8;
    case torch_dtype::int8:
        return torch::kInt8;
    case torch_dtype::int16:
        return torch::kInt16;
    case torch_dtype::_float:
        return torch::kFloat;
    case torch_dtype::_int:
        return torch::kInt;
    }
}

torch_dtype cvt_Dtype2dtype(torch::Dtype dtype)
{
    switch (dtype)
    {
    case torch::kFloat64:
        return torch_dtype::float64;
    case torch::kInt64:
        return torch_dtype::int64;
    case torch::kUInt8:
        return torch_dtype::uint8;
    case torch::kInt8:
        return torch_dtype::int8;
    case torch::kInt16:
        return torch_dtype::int16;
    case torch::kFloat:
        return torch_dtype::_float;
    case torch::kInt:
        return torch_dtype::_int;
    }
}


at::Tensor convert_image_dtype_func(at::Tensor &img, torch_dtype _dtype)
{
    torch::Dtype dtype = cvt_dtype2Dtype(_dtype);
    if (img.scalar_type() == dtype)
    {
        return img;
    }
    if (img.is_floating_point())
    {
        if (torch::tensor(0, torch::dtype(dtype)).is_floating_point())
        {
            return img.to(dtype);
        }
        assert((img.dtype() == torch::kFloat32 && (dtype == torch::kInt32 || dtype == torch::kInt64)) ||
               (img.dtype() == torch::kFloat64 && dtype == torch::kInt64));
        float eps = 1e-3;
        float max_val = float(_max_value(dtype));
        at::Tensor result = img.mul(max_val + 1.0 - eps);
        return result.to(dtype);
    }
    else
    {
        float input_max = float(_max_value(img.scalar_type()));
        if (torch::tensor(0, torch::dtype(dtype)).is_floating_point())
        {
            img = img.to(dtype);
            return img / input_max;
        }
        float output_max = float(_max_value(dtype));
        if (input_max > output_max)
        {
            int64_t factor = int64_t((input_max + 1) / (output_max + 1));
            img = img.div(factor, "floor");
            return img.to(dtype);
        }
        else
        {
            int64_t factor = int64_t((output_max + 1) / (input_max + 1));
            img = img.to(dtype);
            return img * factor;
        }
    }
}