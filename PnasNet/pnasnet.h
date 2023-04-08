// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <cmath>

using Options = torch::nn::Conv2dOptions;

// Progressive Neural Architecture Search

struct SepConvImpl : torch::nn::SequentialImpl {
    //Separable Convolution.'''
	SepConvImpl(int64_t in_planes, int64_t out_planes, int64_t kernel_size, int64_t stride, torch::Device device) {
		auto t = torch::nn::Conv2d(Options(in_planes, out_planes,
                kernel_size).stride(stride)
				   .padding(static_cast<int64_t>(std::floor((kernel_size-1)/2)))
                .bias(false)
				   .groups(in_planes));
		t->to(device);
		push_back(t);
		auto z = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
		z->to(device);
		push_back(z);
	}

    torch::Tensor forward(torch::Tensor x) {
    	return torch::nn::SequentialImpl::forward(x);
    }
};

TORCH_MODULE(SepConv);

struct CellAImpl : torch::nn::SequentialImpl {
	int64_t stride{1};
	SepConv sep_conv1{nullptr};
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};

	CellAImpl(int64_t in_planes, int64_t out_planes, int64_t stride /*1*/, torch::Device device);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(CellA);

struct CellBImpl : public torch::nn::Module {
	int64_t stride{1};
	SepConv sep_conv1{nullptr};
	SepConv sep_conv2{nullptr};
	SepConv sep_conv3{nullptr};
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

	explicit CellBImpl(int64_t in_planes, int64_t out_planes, int64_t stride /*1*/, torch::Device device);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(CellB);


struct PNASNetAImpl : public torch::nn::Module {

	int64_t in_planes;
	std::string cell_type;
	torch::nn::Sequential layer1{nullptr}, layer3{nullptr}, layer5{nullptr};
	torch::nn::Sequential layer2{nullptr}, layer4{nullptr};

	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Linear linear{nullptr};

	explicit PNASNetAImpl(int64_t num_cells, int64_t num_planes, int64_t num_classes, torch::Device device=torch::kCPU);

	torch::nn::Sequential downsample(int64_t planes, torch::Device device);

	torch::Tensor forward(torch::Tensor x);

	torch::nn::Sequential _make_layer(int64_t num_planes, int64_t num_cells, torch::Device device);
};


TORCH_MODULE(PNASNetA);


struct PNASNetBImpl : public torch::nn::Module {

	int64_t in_planes;
	std::string cell_type;
	torch::nn::Sequential layer1{nullptr}, layer3{nullptr}, layer5{nullptr};
	torch::nn::Sequential layer2{nullptr}, layer4{nullptr};

	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Linear linear{nullptr};

	PNASNetBImpl(int64_t num_cells, int64_t num_planes, int64_t num_classes, torch::Device device=torch::kCPU);

	torch::nn::Sequential _make_layer(int64_t num_planes, int64_t num_cells, torch::Device device);
	torch::nn::Sequential downsample(int64_t planes, torch::Device device);

	torch::Tensor forward(torch::Tensor x);
};


TORCH_MODULE(PNASNetB);

//    PNASNetA(CellA, num_cells=6, num_planes=44)
//    PNASNetB(CellB, num_cells=6, num_planes=32)





