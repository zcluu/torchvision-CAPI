#include <functional.h>

at::Tensor rgb_to_grayscale_func(at::Tensor &img, int num_output_channels)
{
    if (img.ndimension() < 3)
    {
        throw std::runtime_error("Input image tensor should have at least 3 dimensions");
    }
    if (img.size(-3) != 1 && img.size(-3) != 3)
    {
        throw std::runtime_error("Input image tensor should have 1 or 3 channels");
    }
    if (num_output_channels != 1 && num_output_channels != 3)
    {
        throw std::invalid_argument("num_output_channels should be either 1 or 3");
    }
    if (img.size(-3) == 3)
    {
        auto rgb = img.unbind(-3);
        auto r = rgb[0];
        auto g = rgb[1];
        auto b = rgb[2];

        auto l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype());
        l_img = l_img.unsqueeze(-3);

        if (num_output_channels == 3)
        {
            return l_img.expand_as(img);
        }
        return l_img;
    }

    if (num_output_channels == 3)
    {
        return img.expand_as(img);
    }
    return img;
}