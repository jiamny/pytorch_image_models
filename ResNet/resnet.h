#pragma once

/*
 Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
 */

#include <torch/torch.h>

struct BasicBlockImpl : public torch::nn::Module {

  int64_t stride{1};
  torch::nn::Sequential downsample;

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

  int64_t expansion{1};
  torch::nn::Sequential shortcut{nullptr};
  bool useShortcut{false};

  BasicBlockImpl(int64_t in_planes, int64_t planes, int64_t stride=1); // stride=1

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BasicBlock);

struct Bottleneck_Impl : public torch::nn::Module {

  int64_t stride{1};
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};

  int expansion{4};
  torch::nn::Sequential shortcut{nullptr};
  bool useShortcut{false};

  Bottleneck_Impl(int64_t in_planes, int64_t planes, int64_t stride=1);

  torch::Tensor forward(torch::Tensor X);
};

TORCH_MODULE(Bottleneck_);


struct ResNetBBImpl : public torch::nn::Module {
  int64_t in_planes{64};
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  std::vector<BasicBlock> layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
  torch::nn::Linear linear{nullptr};
  int64_t expansion{1};

  std::vector<BasicBlock> _make_layer(
          int64_t planes,
          int64_t blocks,
          int64_t stride);

  explicit ResNetBBImpl(std::vector<int> num_blocks, int64_t num_classes);

  torch::Tensor forward(torch::Tensor X);
};

TORCH_MODULE(ResNetBB);


struct ResNetBNImpl : public torch::nn::Module {
  int64_t in_planes{64};
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  std::vector<Bottleneck_> layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
  torch::nn::Linear linear{nullptr};
  int64_t expansion{1};

  std::vector<Bottleneck_> _make_layer(
          int64_t planes,
          int64_t blocks,
          int64_t stride);

  explicit ResNetBNImpl(std::vector<int> num_blocks, int64_t num_classes);

  torch::Tensor forward(torch::Tensor X);
};

TORCH_MODULE(ResNetBN);

ResNetBB ResNet18(int64_t num_classes);
ResNetBB ResNet34(int64_t num_classes);
ResNetBN ResNet50(int64_t num_classes);
ResNetBN ResNet101(int64_t num_classes);
ResNetBN ResNet152(int64_t num_classes);



