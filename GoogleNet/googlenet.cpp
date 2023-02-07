
#include "googlenet.h"
#include <torch/torch.h>

using Options = torch::nn::Conv2dOptions;

InceptionImpl::InceptionImpl(int64_t in_planes, int64_t n1x1, int64_t n3x3red, int64_t n3x3, int64_t n5x5red, int64_t n5x5, int64_t pool_planes){
	//# 1x1 conv branch
    b1 = torch::nn::Sequential(
    		torch::nn::Conv2d(Options(in_planes, n1x1, 1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(n1x1)),
            torch::nn::ReLU(true)
        );

	//# 1x1 conv -> 3x3 conv branch
	b2 = torch::nn::Sequential(
			torch::nn::Conv2d(Options(in_planes, n3x3red, 1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(n3x3red)),
			torch::nn::ReLU(true),
			torch::nn::Conv2d(Options(n3x3red, n3x3, 3).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(n3x3)),
			torch::nn::ReLU(true)
		);

	//# 1x1 conv -> 5x5 conv branch
	b3 = torch::nn::Sequential(
			torch::nn::Conv2d(Options(in_planes, n5x5red, 1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(n5x5red)),
			torch::nn::ReLU(true),
			torch::nn::Conv2d(Options(n5x5red, n5x5, 3).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(n5x5)),
			torch::nn::ReLU(true),
			torch::nn::Conv2d(Options(n5x5, n5x5, 3).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(n5x5)),
			torch::nn::ReLU(true)
		);

	//# 3x3 pool -> 1x1 conv branch
    b4 = torch::nn::Sequential(
    		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(1).padding(1)),
			torch::nn::Conv2d(Options(in_planes, pool_planes, 1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(pool_planes)),
			torch::nn::ReLU(true)
        );

}


torch::Tensor InceptionImpl::forward(torch::Tensor x){
	auto y1 = b1->forward(x);
	auto y2 = b2->forward(x);
	auto y3 = b3->forward(x);
	auto y4 = b4->forward(x);

	//auto cuda_available = torch::cuda::is_available();
	//torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	//auto z = torch::cat({y1,y2,y3,y4}, 1).to(device);
	//std::cout << z.device() << '\n';
	return torch::cat({y1,y2,y3,y4}, 1);
}

GoogleNetImpl::GoogleNetImpl(int64_t num_classes) {
	/*
	this->pre_layers = torch::nn::Sequential(
				torch::nn::Conv2d(Options(3, 192, 3).padding(1)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(192)),
				torch::nn::ReLU(true)
	        );
	*/
	conv1 = torch::nn::Conv2d(Options(3, 192, 3).padding(1));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(192));

	a3 = Inception(192,  64,  96, 128, 16, 32, 32);
	b3 = Inception(256, 128, 128, 192, 32, 96, 64);

	maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

	a4 = Inception(480, 192,  96, 208, 16,  48,  64);
	b4 = Inception(512, 160, 112, 224, 24,  64,  64);
	c4 = Inception(512, 128, 128, 256, 24,  64,  64);
	d4 = Inception(512, 112, 144, 288, 32,  64,  64);
	e4 = Inception(528, 256, 160, 320, 32, 128, 128);

	a5 = Inception(832, 256, 160, 320, 32, 128, 128);
	b5 = Inception(832, 384, 192, 384, 48, 128, 128);

	avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(8).stride(1));
	linear = torch::nn::Linear(1024, num_classes);

	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("a3", a3);
	register_module("b3", b3);
	register_module("a4", a4);
	register_module("b4", b4);
	register_module("c4", c4);
	register_module("d4", d4);
	register_module("e4", e4);
	register_module("a5", a5);
	register_module("b5", b5);
	register_module("linear", linear);

    // Initializing weights
    for (auto& module : modules(/*include_self=*/false)) {
    	if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
    		torch::nn::init::kaiming_normal_(
            M->weight,
            /*a=*/0,
            torch::kFanOut,
            torch::kReLU);
    	else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
    		torch::nn::init::constant_(M->weight, 1);
    		torch::nn::init::constant_(M->bias, 0);
        } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
    		torch::nn::init::normal_(M->weight, 0.0, 0.01);
    		torch::nn::init::constant_(M->bias, 0);
        }
    }
}


torch::Tensor GoogleNetImpl::forward(torch::Tensor x){
    //auto out = pre_layers->forward(x);
	x = torch::relu(bn1->forward(conv1->forward(x)));
    x = a3->forward(x);
    x = b3->forward(x);
    x = maxpool->forward(x);
    x = a4->forward(x);
    x = b4->forward(x);
    x = c4->forward(x);
    x = d4->forward(x);
    x = e4->forward(x);
    x = maxpool->forward(x);
    x = a5->forward(x);
    x = b5->forward(x);
    x = avgpool->forward(x);
    x = x.view({x.size(0), -1});
    x = linear->forward(x);
    return x;
}
