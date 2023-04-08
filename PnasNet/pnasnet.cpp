
#include "pnasnet.h"
#include <torch/torch.h>

CellAImpl::CellAImpl(int64_t in_planes, int64_t out_planes, int64_t stride_ , torch::Device device) {
	stride = stride_;
	sep_conv1 = SepConv(in_planes, out_planes, 7, stride, device);

	if( stride==2 ){
		conv1 = torch::nn::Conv2d(Options(in_planes, out_planes, 1).stride(1).padding(0).bias(false));
		bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
		conv1->to(device);
		bn1->to(device);

		register_module("conv1", conv1);
		register_module("bn1", bn1);
	}

	register_module("sep_conv1", sep_conv1);
}

torch::Tensor CellAImpl::forward(torch::Tensor x){
	auto y1 = sep_conv1->forward(x);
	auto y2 = torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(3).stride(stride).padding(1));
	if( stride==2 )
		y2 = bn1->forward(conv1->forward(y2));
	return torch::relu(y1+y2);
}



CellBImpl::CellBImpl(int64_t in_planes, int64_t out_planes, int64_t stride_, torch::Device device){
	stride = stride_;

	//Left branch
	sep_conv1 = SepConv(in_planes, out_planes, 7, stride, device);
	sep_conv2 = SepConv(in_planes, out_planes, 3, stride, device);

    //Right branch
	sep_conv3 = SepConv(in_planes, out_planes, 5, stride, device);

	if( stride==2 ) {
		conv1 = torch::nn::Conv2d(Options(in_planes, out_planes, 1).stride(1).padding(0).bias(false));
	    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
	    conv1->to(device);
	    bn1->to(device);

		register_module("conv1", conv1);
		register_module("bn1", bn1);
	}

	//Reduce channels
	conv2 = torch::nn::Conv2d(Options(2*out_planes, out_planes, 1).stride(1).padding(0).bias(false));
	bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
	conv2->to(device);
	bn2->to(device);

	register_module("sep_conv1", sep_conv1);
	register_module("sep_conv2", sep_conv2);
	register_module("conv2", conv2);
	register_module("bn2", bn2);
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


PNASNetAImpl::PNASNetAImpl(int64_t num_cells, int64_t num_planes, int64_t num_classes, torch::Device device) {
	in_planes = num_planes;

	conv1 = torch::nn::Conv2d(Options(3, num_planes, 3).stride(1).padding(1).bias(false));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_planes));
	conv1->to(device);
	bn1->to(device);

	layer1 = _make_layer(num_planes, 6, device);
	layer2 = downsample(num_planes*2, device);
	layer3 = _make_layer(num_planes*2, 6, device);
	layer4 = downsample(num_planes*4, device);
	layer5 = _make_layer(num_planes*4, 6, device);

	linear = torch::nn::Linear(num_planes*4, num_classes);
	linear->to(device);

	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
	register_module("layer5", layer5);
	register_module("linear", linear);
}

torch::nn::Sequential PNASNetAImpl::downsample(int64_t planes, torch::Device device) {
	torch::nn::Sequential layer;
	layer->push_back( CellA(in_planes, planes, 2, device) );
	in_planes = planes;
	return layer;
}


torch::nn::Sequential PNASNetAImpl::_make_layer(int64_t planes, int64_t num_cells, torch::Device device) {
	torch::nn::Sequential layers;

	for( int i = 0; i < num_cells; i++ ) {
		layers->push_back(CellA(in_planes, planes, 1, device));
	    in_planes = planes;
	}

	return layers;
}

torch::Tensor PNASNetAImpl::forward(torch::Tensor x) {
	x = torch::relu(bn1->forward(conv1->forward(x)));

	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);
	x = layer5->forward(x);

	x = torch::nn::functional::avg_pool2d(x, torch::nn::functional::AvgPool2dFuncOptions(8));
	x = linear->forward(x.view({x.size(0), -1}));

	return x;
}


PNASNetBImpl::PNASNetBImpl(int64_t num_cells, int64_t num_planes, int64_t num_classes, torch::Device device) {
	in_planes = num_planes;

	conv1 = torch::nn::Conv2d(Options(3, num_planes, 3).stride(1).padding(1).bias(false));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_planes));
	conv1->to(device);
	bn1->to(device);

	layer1 = _make_layer(num_planes, 6, device);
	layer2 = downsample(num_planes*2, device);
	layer3 = _make_layer(num_planes*2, 6, device);
	layer4 = downsample(num_planes*4, device);
	layer5 = _make_layer(num_planes*4, 6, device);

	linear = torch::nn::Linear(num_planes*4, num_classes);
	linear->to(device);

	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
	register_module("layer5", layer5);
	register_module("linear", linear);
}



torch::nn::Sequential PNASNetBImpl::downsample(int64_t planes, torch::Device device) {
	torch::nn::Sequential layer;
	layer->push_back( CellB(in_planes, planes, 2, device) );
	in_planes = planes;
	return layer;
}

torch::nn::Sequential PNASNetBImpl::_make_layer(int64_t planes, int64_t num_cells, torch::Device device) {
	torch::nn::Sequential layers;

	for( int i = 0; i < num_cells; i++ ) {
		layers->push_back(CellB(in_planes, planes, 1, device));
	    in_planes = planes;
	}

	return layers;
}


torch::Tensor PNASNetBImpl::forward(torch::Tensor x) {
	x = torch::relu(bn1->forward(conv1->forward(x)));
//	std::cout << out.sizes() << std::endl;
	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);
	x = layer5->forward(x);

	x = torch::nn::functional::avg_pool2d(x, torch::nn::functional::AvgPool2dFuncOptions(8));
	x = linear->forward(x.view({x.size(0), -1}));

	return x;
}

