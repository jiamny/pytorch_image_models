#pragma once

/*
 Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
 */

#include <torch/torch.h>

using Options = torch::nn::Conv2dOptions;

struct BasicBlock : public torch::nn::Module {

  int64_t stride{1};

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

  int64_t expansion{1};
  torch::nn::Sequential shortcut;

  BasicBlock(int64_t in_planes, int64_t planes, int64_t stride_=1);

  torch::Tensor forward(torch::Tensor x);

};


struct Bottleneck : public torch::nn::Module {

  int64_t stride{1};
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};

  int expansion{4};
  torch::nn::Sequential shortcut{nullptr};

  Bottleneck(int64_t in_planes, int64_t planes, int64_t stride_=1);

  torch::Tensor forward(torch::Tensor x);
};


struct ResNetBB : public torch::nn::Module {
  int64_t in_planes{64};
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
  torch::nn::Linear linear{nullptr};
  int64_t expansion{1};

  ResNetBB(std::vector<int> num_blocks, int64_t num_classes);

  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Sequential _make_layer(
          int64_t planes,
          int64_t blocks,
          int64_t stride = 1);
};


struct ResNetBN : public torch::nn::Module {
  int64_t in_planes{64};
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
  torch::nn::Linear linear{nullptr};
  int64_t expansion{4};

  ResNetBN(std::vector<int> num_blocks, int64_t num_classes);

  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Sequential _make_layer(
          int64_t planes,
          int64_t blocks,
          int64_t stride = 1);

};


ResNetBB ResNet18(int64_t num_classes);

ResNetBB ResNet34(int64_t num_classes);

ResNetBN ResNet50(int64_t num_classes);

ResNetBN ResNet101(int64_t num_classes);


