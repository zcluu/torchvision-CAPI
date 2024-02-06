// #include <functional.h>
// #include <utils.h>

// at::Tensor pad_func(at::Tensor& tensor, ImagePadding_t padding, int64_t fill, at::padding_mode padding_mode) {
//     bool need_squeeze = false;
//     if (tensor.ndimension() < 4)
//     {
//         tensor = tensor.unsqueeze(0);
//         need_squeeze = true;
//     }

//     at::TensorOptions out_dtype = tensor.dtype();
//     bool need_cast = false;
//     if (at::padding_mode_string(padding_mode) == "constant" && tensor.dtype() != torch::kFloat32 && tensor.dtype() != torch::kFloat64) {
//         need_cast = true;
//         tensor = tensor.to(torch::kFloat);
//     }
//     {
//         /* code */
//     }


// }