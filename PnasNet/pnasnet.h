// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <cmath>

// Progressive Neural Architecture Search

struct SepConvImpl : torch::nn::Module {
    //Separable Convolution.'''
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};

	explicit SepConvImpl(int64_t in_planes, int64_t out_planes, int64_t kernel_size, int64_t stride);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(SepConv);

struct CellAImpl : torch::nn::Module {
	int64_t stride{1};
	SepConv sep_conv1{nullptr};
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};

	explicit CellAImpl(int64_t in_planes, int64_t out_planes, int64_t stride /*1*/);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(CellA);

struct CellBImpl : torch::nn::Module {
	int64_t stride{1};
	SepConv sep_conv1{nullptr};
	SepConv sep_conv2{nullptr};
	SepConv sep_conv3{nullptr};
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

	explicit CellBImpl(int64_t in_planes, int64_t out_planes, int64_t stride /*1*/);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(CellB);


struct PNASNetAImpl : torch::nn::Module {

	int64_t in_planes;
	std::string cell_type;
	std::vector<CellA> layer1, layer3, layer5;
	CellA layer2{nullptr}, layer4{nullptr};

	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Sequential linear;

	explicit PNASNetAImpl(int64_t num_cells, int64_t num_planes, int64_t num_classes);

	std::vector<CellA> _make_layer(int64_t num_planes, int64_t num_cells);
	CellA downsample(int64_t planes);

	torch::Tensor forward(torch::Tensor x);
};


TORCH_MODULE(PNASNetA);


struct PNASNetBImpl : torch::nn::Module {

	int64_t in_planes;
	std::string cell_type;
	std::vector<CellB> layer1, layer3, layer5;
	CellB layer2{nullptr}, layer4{nullptr};

	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Sequential linear;

	explicit PNASNetBImpl(int64_t num_cells, int64_t num_planes, int64_t num_classes);

	std::vector<CellB> _make_layer(int64_t num_planes, int64_t num_cells);
	CellB downsample(int64_t planes);

	torch::Tensor forward(torch::Tensor x);
};


TORCH_MODULE(PNASNetB);

//    PNASNetA(CellA, num_cells=6, num_planes=44)
//    PNASNetB(CellB, num_cells=6, num_planes=32)





