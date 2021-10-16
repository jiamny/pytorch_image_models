#pragma once

#include <torch/torch.h>
#include <map>

struct BottleneckImpl : public torch::nn::Module {
	int64_t out_planes;
	int64_t dense_depth;
	bool first_layer{false};
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
	torch::nn::Sequential shortcut;

	explicit BottleneckImpl(int64_t last_planes, int64_t in_planes, int64_t out_planes,
			int64_t dense_depth, int64_t stride, bool first_layer);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Bottleneck);

struct DPNImpl : public torch::nn::Module {
	std::vector<int64_t> in_planes, out_planes, num_blocks, dense_depth;
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	std::vector<Bottleneck> layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
	torch::nn::Sequential linear;
	int64_t last_planes;

	explicit DPNImpl(std::map<std::string, std::vector<int64_t>> cfg, int64_t num_classes);

	std::vector<Bottleneck> _make_layer(int64_t in_planes, int64_t out_planes, int64_t num_blocks, int64_t dense_depth, int64_t stride);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(DPN);

DPN DPN26(int64_t num_classes);
DPN DPN92(int64_t num_classes);



