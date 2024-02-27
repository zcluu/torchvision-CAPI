#include <functional.h>

at::Tensor crop_func(at::Tensor &img, int64_t top, int64_t left, int64_t height, int64_t width)
{
    ImageSize_t image_size = get_image_size(img);
    int64_t h = image_size.first;
    int64_t w = image_size.second;
    // std::cout << "crop_func: " << top << " " << left << " " << height << " " << width << std::endl;
    int64_t right = left + width;
    int64_t bottom = top + height;
    // std::cout << "crop_func: right:" << right << " bottom:" << bottom << std::endl;
    // std::cout << "crop_func: h:" << h << " w:" << w << std::endl;
    if (left < 0 || top < 0 || right > w || bottom > h)
    {
        std::vector<int64_t> padding_lrtb = {
            max(-left + min(0, right), 0),
            max(right - max(w, left), 0),
            max(-top + min(0, bottom), 0),
            max(bottom - max(h, top), 0),
        };
        torch::Tensor padded_img = pad(img.index({"...",
                                                  torch::indexing::Slice(max(top, 0LL), bottom),
                                                  torch::indexing::Slice(max(left, 0LL), right)}),
                                       padding_lrtb, "constant", 0);

        return padded_img;
    }
    return img.index({"...",
                      torch::indexing::Slice(top, bottom),
                      torch::indexing::Slice(left, right)});
}
at::Tensor center_crop_func(at::Tensor &img, ImageSize_t output_size)
{
    ImageDims_t dims = get_image_dims(img);
    int64_t image_height = dims.height;
    int64_t image_width = dims.width;
    int64_t crop_height = output_size.first;
    int64_t crop_width = output_size.second;

    if (crop_width > image_width || crop_height > image_height)
    {
        std::vector<int64_t> padding_ltrb = {
            crop_width > image_width ? (crop_width - image_width) / 2 : 0,
            crop_height > image_height ? (crop_height - image_height) / 2 : 0,
            crop_width > image_width ? (crop_width - image_width + 1) / 2 : 0,
            crop_height > image_height ? (crop_height - image_height + 1) / 2 : 0};
        img = torch::pad(img, padding_ltrb, "constant", 0);
        dims = get_image_dims(img);
        image_height = dims.height;
        image_width = dims.width;
        if (crop_width == image_width && crop_height == image_height)
        {
            return img;
        }
    }
    int64_t crop_top = (image_height - crop_height) / 2;
    int64_t crop_left = (image_width - crop_width) / 2;
    return crop_func(img, crop_top, crop_left, crop_height, crop_width);
}
at::Tensor resized_crop_func(
    at::Tensor &img,
    int64_t top,
    int64_t left,
    int64_t height,
    int64_t width,
    ImageSize_t resize)
{
    img = crop_func(img, top, left, height, width);
    // img = resize_func(img, resize);
    return img;
}