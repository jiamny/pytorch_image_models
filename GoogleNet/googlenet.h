// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <cmath>

struct InceptionImpl : public torch::nn::Module {

	torch::nn::Sequential b1{nullptr}, b2{nullptr}, b3{nullptr}, b4{nullptr};

	explicit InceptionImpl(int64_t in_planes, int64_t n1x1, int64_t n3x3red, int64_t n3x3,
							int64_t n5x5red, int64_t n5x5, int64_t pool_planes, torch::Device device);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Inception);


struct GoogleNetImpl : public torch::nn::Module {

	//torch::nn::Sequential pre_layers;
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};

	Inception a3{nullptr}, b3{nullptr}, a4{nullptr}, b4{nullptr}, c4{nullptr}, d4{nullptr}, e4{nullptr}, a5{nullptr}, b5{nullptr};

	torch::nn::AvgPool2d avgpool{nullptr};
	torch::nn::MaxPool2d maxpool{nullptr};
	torch::nn::Linear linear{nullptr};

	explicit GoogleNetImpl(int64_t num_classes, torch::Device device=torch::kCPU);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(GoogleNet);







