#pragma once

#include <torch/torch.h>

// Densenet-BC model class, based on
// "Densely Connected Convolutional Networks"
// <https://arxiv.org/pdf/1608.06993.pdf>

struct BottleneckImpl : public torch::nn::Module {
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};

	explicit BottleneckImpl(int64_t in_planes, int64_t growth_rate);
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Bottleneck);

struct TransitionImpl : public torch::nn::Module {
	torch::nn::BatchNorm2d bn{nullptr};
	torch::nn::Conv2d conv{nullptr};

	explicit TransitionImpl(int64_t in_planes, int64_t out_planes);
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Transition);

struct DenseNetImpl : public torch::nn::Module {
	torch::nn::BatchNorm2d bn{nullptr};
	torch::nn::Conv2d conv1{nullptr};
	std::vector<Bottleneck> dense1, dense2, dense3, dense4;
	Transition trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
	int64_t growth_rate;
	torch::nn::Linear linear{nullptr};

	explicit DenseNetImpl(std::vector<int64_t> nblocks, int64_t growth_rate, double reduction, int64_t num_classes);
	torch::Tensor forward(torch::Tensor x);
	std::vector<Bottleneck> _make_dense_layers(int64_t in_planes, int64_t nblock);
};

TORCH_MODULE(DenseNet);



