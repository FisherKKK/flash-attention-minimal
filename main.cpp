#include <torch/extension.h>

// 首先这边相当于是说先写一个forward接口, 然后对应的Python Module会进行调用
torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}