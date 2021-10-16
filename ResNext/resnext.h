#pragma once

/*
 See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
 */

#include <torch/torch.h>

struct ResNextBlockImpl : public torch::nn::Module {
	// Grouped convolution block.
	int64_t expansion = 2, cardinality=32, bottleneck_width=4, stride = 1;

	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};

	torch::nn::Sequential shortcut{nullptr};
	bool useShortcut = false;

	ResNextBlockImpl(int64_t in_planes, int64_t cardinality, int64_t bottleneck_width, int64_t stride); // stride=1

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ResNextBlock);


struct ResNextImpl : public torch::nn::Module {
  int64_t in_planes = 64, cardinality = 0, bottleneck_width = 0, num_classes = 10;
  std::vector<int> num_blocks;
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  std::vector<ResNextBlock> layer1{nullptr}, layer2{nullptr}, layer3{nullptr};
  torch::nn::Linear linear{nullptr};
  int64_t expansion = 2;

  std::vector<ResNextBlock> _make_layer(
          int64_t blocks,
          int64_t stride);

  explicit ResNextImpl(std::vector<int> num_blocks, int64_t cardinality, int64_t bottleneck_width, int64_t num_classes);

  torch::Tensor forward(torch::Tensor X);
};

TORCH_MODULE(ResNext);



ResNext ResNeXt29_2x64d(int64_t num_classes);
ResNext ResNeXt29_4x64d(int64_t num_classes);
ResNext ResNeXt29_8x64d(int64_t num_classes);
ResNext ResNeXt29_32x4d(int64_t num_classes);




