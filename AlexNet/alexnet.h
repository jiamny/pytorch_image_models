#pragma once

#include <torch/torch.h>

// AlexNet model architecture from the
// "One weird trick..." <https://arxiv.org/abs/1404.5997> paper.
struct AlexNetImpl : public torch::nn::Module {
  torch::nn::Sequential features{nullptr}, classifier{nullptr};

  explicit AlexNetImpl(int64_t num_classes = 10);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(AlexNet);
