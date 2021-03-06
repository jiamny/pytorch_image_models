#pragma once

#include <torch/torch.h>

struct VGGImpl : public torch::nn::Module {
  torch::nn::Sequential features{nullptr}, classifier{nullptr};

  void _initialize_weights();

  explicit VGGImpl(
      const torch::nn::Sequential& features,
      int64_t num_classes = 1000,
      bool initialize_weights = true);

  torch::Tensor forward(torch::Tensor x);
};

// VGG 11-layer model (configuration "A")
struct VGG11Impl : public VGGImpl {
  explicit VGG11Impl(
      int64_t num_classes = 1000,
      bool initialize_weights = true);
};

// VGG 13-layer model (configuration "B")
struct VGG13Impl : public VGGImpl {
  explicit VGG13Impl(
      int64_t num_classes = 1000,
      bool initialize_weights = true);
};

// VGG 16-layer model (configuration "D")
struct VGG16Impl : public VGGImpl {
  explicit VGG16Impl(
      int64_t num_classes = 1000,
      bool initialize_weights = true);
};

// VGG 19-layer model (configuration "E")
struct VGG19Impl : public VGGImpl {
  explicit VGG19Impl(
      int64_t num_classes = 1000,
      bool initialize_weights = true);
};

// VGG 11-layer model (configuration "A") with batch normalization
struct VGG11BNImpl : public VGGImpl {
  explicit VGG11BNImpl(
      int64_t num_classes = 1000,
      bool initialize_weights = true);
};

// VGG 13-layer model (configuration "B") with batch normalization
struct VGG13BNImpl : public VGGImpl {
  explicit VGG13BNImpl(
      int64_t num_classes = 1000,
      bool initialize_weights = true);
};

// VGG 16-layer model (configuration "D") with batch normalization
struct VGG16BNImpl : public VGGImpl {
  explicit VGG16BNImpl(
      int64_t num_classes = 1000,
      bool initialize_weights = true);
};

// VGG 19-layer model (configuration 'E') with batch normalization
struct VGG19BNImpl : public VGGImpl {
  explicit VGG19BNImpl(
      int64_t num_classes = 1000,
      bool initialize_weights = true);
};

TORCH_MODULE(VGG);

TORCH_MODULE(VGG11);
TORCH_MODULE(VGG13);
TORCH_MODULE(VGG16);
TORCH_MODULE(VGG19);

TORCH_MODULE(VGG11BN);
TORCH_MODULE(VGG13BN);
TORCH_MODULE(VGG16BN);
TORCH_MODULE(VGG19BN);

