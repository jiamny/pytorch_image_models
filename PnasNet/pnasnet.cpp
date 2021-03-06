
#include "pnasnet.h"
#include <torch/torch.h>

using Options = torch::nn::Conv2dOptions;

SepConvImpl::SepConvImpl(int64_t in_planes, int64_t out_planes, int64_t kernel_size, int64_t stride) {

	this->conv1 = torch::nn::Conv2d(Options(in_planes, out_planes,
	    	                               kernel_size).stride(stride)
										   .padding(static_cast<int64_t>(std::floor((kernel_size-1)/2)))
	    	                               .bias(false)
										   .groups(in_planes));

	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
}

torch::Tensor SepConvImpl::forward(torch::Tensor x){
	auto out = bn1->forward(conv1->forward(x));
//	std::cout << " +++ " << out.sizes() << std::endl;
	return out;
}

CellAImpl::CellAImpl(int64_t in_planes, int64_t out_planes, int64_t stride ) {
	this->stride = stride;
	this->sep_conv1 = SepConv(in_planes, out_planes, 7, stride);

	if( stride==2 ){
		this->conv1 = torch::nn::Conv2d(Options(in_planes, out_planes, 1).stride(1).padding(0).bias(false));
		this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
	}
}

torch::Tensor CellAImpl::forward(torch::Tensor x){
	auto y1 = sep_conv1->forward(x);
	auto y2 = torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(3).stride(stride).padding(1));
	if( this->stride==2 )
		y2 = bn1->forward(conv1->forward(y2));
	return torch::relu(y1+y2);
}

CellBImpl::CellBImpl(int64_t in_planes, int64_t out_planes, int64_t stride ){
	this->stride = stride;

	//Left branch
	this->sep_conv1 = SepConv(in_planes, out_planes, 7, stride);
	this->sep_conv2 = SepConv(in_planes, out_planes, 3, stride);

    //Right branch
	this->sep_conv3 = SepConv(in_planes, out_planes, 5, stride);

	if( stride==2 ) {
		this->conv1 = torch::nn::Conv2d(Options(in_planes, out_planes, 1).stride(1).padding(0).bias(false));
	    this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
	}

	//Reduce channels
	this->conv2 = torch::nn::Conv2d(Options(2*out_planes, out_planes, 1).stride(1).padding(0).bias(false));
	this->bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
}

torch::Tensor CellBImpl::forward(torch::Tensor x){
	//Left branch
	auto y1 = sep_conv1->forward(x);
//	std::cout << "y1 -> " << y1.sizes() << std::endl;
	auto y2 = sep_conv2->forward(x);
//	std::cout << "y2 -> " << y2.sizes() << std::endl;

	//Right branch
	auto y3 = torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(3).stride(stride).padding(1));
//	std::cout << "y3-1 -> " << y3.sizes() << std::endl;
	if( stride==2 ) {
		y3 = bn1->forward(conv1->forward(y3));
//		std::cout << "y3-2 -> " << y3.sizes() << std::endl;
	}

	auto y4 = sep_conv3->forward(x);
//	std::cout << "y4 -> " << y4.sizes() << std::endl;

	// Concat & reduce channels
	auto b1 = torch::relu(y1+y2);
//	std::cout << "b1 -> " << b1.sizes() << std::endl;
	auto b2 = torch::relu(y3+y4);
//	std::cout << "b2 -> " << b2.sizes() << std::endl;
    auto y = torch::cat({b1,b2}, 1);
//    std::cout << "y -> " << y.sizes() << std::endl;

  return torch::relu(bn2->forward(conv2->forward(y)));
}


PNASNetAImpl::PNASNetAImpl(int64_t num_cells, int64_t num_planes, int64_t num_classes) {
	this->in_planes = num_planes;

	this->conv1 = torch::nn::Conv2d(Options(3, num_planes, 3).stride(1).padding(1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_planes));

	this->layer1 = _make_layer(num_planes, 6);
	this->layer2 = downsample(num_planes*2);
	this->layer3 = _make_layer(num_planes*2, 6);
	this->layer4 = downsample(num_planes*4);
	this->layer5 = _make_layer(num_planes*4, 6);

	this->linear->push_back(torch::nn::Linear(num_planes*4, num_classes));
}

CellA PNASNetAImpl::downsample(int64_t planes) {
	auto layer = CellA(this->in_planes, planes, 2);
	this->in_planes = planes;
	return layer;
}


std::vector<CellA> PNASNetAImpl::_make_layer(int64_t planes, int64_t num_cells) {
	std::vector<CellA> layers;

	for( int i = 0; i < num_cells; i++ ) {
		auto layer = CellA(this->in_planes, planes, 1);
		layers.push_back(layer);
	    this->in_planes = planes;
	}

	return layers;
}

torch::Tensor PNASNetAImpl::forward(torch::Tensor x) {
	auto out = torch::relu(bn1->forward(conv1->forward(x)));
//	std::cout << out.sizes() << std::endl;

	for( int i =0; i < layer1.size(); i++ )
		out = layer1[i]->forward(out);

	out = layer2->forward(out);

	for( int i =0; i < layer3.size(); i++ )
		out = layer3[i]->forward(out);

	out = layer4->forward(out);

	for( int i =0; i < layer5.size(); i++ )
		out = layer5[i]->forward(out);

	out = torch::nn::functional::avg_pool2d(out, torch::nn::functional::AvgPool2dFuncOptions(8));
	out = linear->forward(out.view({out.size(0), -1}));

	return out;
}


PNASNetBImpl::PNASNetBImpl(int64_t num_cells, int64_t num_planes, int64_t num_classes) {
	this->in_planes = num_planes;

	this->conv1 = torch::nn::Conv2d(Options(3, num_planes, 3).stride(1).padding(1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_planes));

	this->layer1 = _make_layer(num_planes, 6);
	this->layer2 = downsample(num_planes*2);
	this->layer3 = _make_layer(num_planes*2, 6);
	this->layer4 = downsample(num_planes*4);
	this->layer5 = _make_layer(num_planes*4, 6);

	this->linear->push_back(torch::nn::Linear(num_planes*4, num_classes));
}



CellB PNASNetBImpl::downsample(int64_t planes) {
	auto layer = CellB(this->in_planes, planes, 2);
	this->in_planes = planes;
	return layer;
}

std::vector<CellB> PNASNetBImpl::_make_layer(int64_t planes, int64_t num_cells) {
	std::vector<CellB> layers;

	for( int i = 0; i < num_cells; i++ ) {
		auto layer = CellB(this->in_planes, planes, 1);
		layers.push_back(layer);
	    this->in_planes = planes;
	}

	return layers;
}


torch::Tensor PNASNetBImpl::forward(torch::Tensor x) {
	auto out = torch::relu(bn1->forward(conv1->forward(x)));
//	std::cout << out.sizes() << std::endl;

//	std::cout << "----" << std::endl;
	for( int i =0; i < layer1.size(); i++ ) {
		out = layer1[i]->forward(out);
//		std::cout << "i = " << i << " " << out.sizes() << std::endl;
	}

	out = layer2->forward(out);

	for( int i =0; i < layer3.size(); i++ )
		out = layer3[i]->forward(out);

	out = layer4->forward(out);

	for( int i =0; i < layer5.size(); i++ )
		out = layer5[i]->forward(out);

	out = torch::nn::functional::avg_pool2d(out, torch::nn::functional::AvgPool2dFuncOptions(8));
	out = linear->forward(out.view({out.size(0), -1}));

	return out;
}

