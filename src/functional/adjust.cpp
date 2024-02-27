#include <functional.h>

at::Tensor adjust_brightness_func(at::Tensor &img, float brightness_factor)
{
    assert(brightness_factor >= 0);
    at::Tensor zero_tensor = torch::zeros_like(img);
    return blend(img, zero_tensor, brightness_factor);
}
at::Tensor adjust_contrast_func(at::Tensor &img, float contrast_factor)
{
    if (contrast_factor < 0)
    {
        throw std::invalid_argument("contrast_factor should be non-negative.");
    }
    ImageDims_t dims = get_image_dims(img);
    int64_t c = dims.channel;
    at::Tensor mean;
    if (c == 3)
    {
        mean = torch::mean(rgb_to_grayscale_func(img).to(img.options()), {-3, -2, -1}, true);
    }
    else
    {
        mean = torch::mean(img, {-3, -2, -1}, true);
    }
    return blend(img, mean, contrast_factor);
}
at::Tensor adjust_saturation_func(at::Tensor &img, float saturation_factor)
{
    if (saturation_factor < 0)
    {
        throw std::invalid_argument("saturation_factor should be non-negative.");
    }

    if (get_image_dims(img).channel == 1)
    {
        return img;
    }
    auto gray_img = rgb_to_grayscale_func(img, 1);
    return blend(img, gray_img, saturation_factor);
}
// at::Tensor adjust_hue_func(at::Tensor &img, float hue_factor)
// {
//     if (hue_factor < -0.5 || hue_factor > 0.5)
//     {
//         throw std::invalid_argument("hue_factor should be in [-0.5, 0.5].");
//     }
//     if (get_image_dims(img).channel == 1)
//     {
//         return img;
//     }
//     torch_dtype orig_dtype = cvt_Dtype2dtype(img.dtype().toScalarType());
//     img = convert_image_dtype_func(img, torch_dtype::float32);

//     img = _rgb2hsv(img);
//     auto channels = torch::unbind(img, -3);
//     auto h = channels[0];
//     auto s = channels[1];
//     auto v = channels[2];
//     h = (h + hue_factor) % 1.0;

// }

// torch::Tensor _rgb2hsv(torch::Tensor img)
// {
//     auto channels = img.unbind(-3);
//     auto r = channels[0];
//     auto g = channels[1];
//     auto b = channels[2];

//     auto img_max = torch::max(img, -3);
//     auto img_min = torch::min(img, -3);
//     torch::Tensor maxc = std::get<0>(img_max);
//     torch::Tensor minc = std::get<0>(img_min);
//     auto eqc = maxc == minc;

//     auto cr = maxc - minc;
//     auto ones = torch::ones_like(maxc);
//     auto s = cr / torch::where(eqc, ones, maxc);
//     auto cr_divisor = torch::where(eqc, ones, cr);
//     auto rc = (maxc - r) / cr_divisor;
//     auto gc = (maxc - g) / cr_divisor;
//     auto bc = (maxc - b) / cr_divisor;
//     auto hr = (maxc == r).to(torch::kFloat32) * (bc - gc);
//     auto hg = ((maxc == g) & (maxc != r)).to(torch::kFloat32) * (2.0 + rc - bc);
//     auto hb = ((maxc != g) & (maxc != r)).to(torch::kFloat32) * (4.0 + gc - rc);
//     auto h = hr + hg + hb;
//     h = torch::fmod((h / 6.0 + 1.0), 1.0);

//     torch::Tensor result = torch::stack({h, s, maxc}, -3);
//     return result;
// }

at::Tensor blend(at::Tensor &img1, at::Tensor &img2, float ratio)
{
    float bound = _max_value(torch::typeMetaToScalarType(img1.dtype()));
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.options());
}