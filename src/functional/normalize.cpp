#include <functional.h>
#include <torch/torch.h>

at::Tensor normalize_func(at::Tensor& tensor, std::vector<float>& mean, std::vector<float>& std, bool inplace) {
    assert(tensor.is_floating_point());
    assert(tensor.ndimension() >= 3);

    if (!inplace) {
        tensor = tensor.clone();
    }

    at::Tensor mean_tensor = torch::from_blob(mean.data(), mean.size(), tensor.options().dtype());
    at::Tensor std_tensor = torch::from_blob(std.data(), std.size(), tensor.options().dtype());

    assert((std_tensor == 0).any().sum().item<int>() == 0);

    if (mean_tensor.ndimension() == 1) {
        mean_tensor = mean_tensor.view({ -1, 1, 1 });
    }

    if (std_tensor.ndimension() == 1) {
        std_tensor = std_tensor.view({ -1, 1, 1 });
    }

    return tensor.sub_(mean_tensor).div_(std_tensor);
}