#include <functional.h>

at::Tensor hflip_func(at::Tensor &img)
{
    return img.flip(-1);
}
at::Tensor vflip_func(at::Tensor &img)
{
    return img.flip(-2);
}