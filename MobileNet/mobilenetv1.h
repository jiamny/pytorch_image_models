#pragma once

#include <torch/torch.h>

struct MobileBlockImpl : public torch::nn::Module {
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::Conv2d conv2{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::BatchNorm2d bn2{nullptr};

    //'''Depthwise conv + Pointwise conv'''
	MobileBlockImpl(int64_t in_planes, int64_t out_planes, int64_t stride=1);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MobileBlock);

struct MobileNetV1Impl : public torch::nn::Module {
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Linear linear{nullptr};
	torch::nn::Sequential layers{nullptr};
	std::vector<std::vector<int64_t>> cfg = {{64, 1}, {128,2}, {128,1}, {256,2}, {256,1}, {512,2}, {512,1}, {512,1}, {512,1}, {512,1}, {512,1}, {1024,2}, {1024,1}};

    explicit MobileNetV1Impl(int64_t num_classes);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential _make_layers(int64_t in_planes);
};

TORCH_MODULE(MobileNetV1);
