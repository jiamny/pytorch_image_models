
#include "googlenet.h"
#include <torch/torch.h>

using Options = torch::nn::Conv2dOptions;

InceptionImpl::InceptionImpl(int64_t in_planes, int64_t n1x1, int64_t n3x3red, int64_t n3x3, int64_t n5x5red, int64_t n5x5, int64_t pool_planes){
	//# 1x1 conv branch
    this->b1 = torch::nn::Sequential(
    		torch::nn::Conv2d(Options(in_planes, n1x1, 1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(n1x1)),
            torch::nn::ReLU(true)
        );

	//# 1x1 conv -> 3x3 conv branch
	this->b2 = torch::nn::Sequential(
			torch::nn::Conv2d(Options(in_planes, n3x3red, 1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(n3x3red)),
			torch::nn::ReLU(true),
			torch::nn::Conv2d(Options(n3x3red, n3x3, 3).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(n3x3)),
			torch::nn::ReLU(true)
		);

	//# 1x1 conv -> 5x5 conv branch
	this->b3 = torch::nn::Sequential(
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
    this->b4 = torch::nn::Sequential(
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
	return torch::cat({y1,y2,y3,y4}, 1);
}

GoogleNetImpl::GoogleNetImpl(int64_t num_classes) {
	this->pre_layers = torch::nn::Sequential(
				torch::nn::Conv2d(Options(3, 192, 3).padding(1)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(192)),
				torch::nn::ReLU(true)
	        );

	this->a3 = Inception(192,  64,  96, 128, 16, 32, 32);
	this->b3 = Inception(256, 128, 128, 192, 32, 96, 64);

	this->maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

	this->a4 = Inception(480, 192,  96, 208, 16,  48,  64);
	this->b4 = Inception(512, 160, 112, 224, 24,  64,  64);
	this->c4 = Inception(512, 128, 128, 256, 24,  64,  64);
	this->d4 = Inception(512, 112, 144, 288, 32,  64,  64);
	this->e4 = Inception(528, 256, 160, 320, 32, 128, 128);

	this->a5 = Inception(832, 256, 160, 320, 32, 128, 128);
	this->b5 = Inception(832, 384, 192, 384, 48, 128, 128);

	this->avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(8).stride(1));
	this->linear = torch::nn::Linear(1024, num_classes);
}


torch::Tensor GoogleNetImpl::forward(torch::Tensor x){
    auto out = pre_layers->forward(x);
    out = a3->forward(out);
    out = b3->forward(out);
    out = maxpool->forward(out);
    out = a4->forward(out);
    out = b4->forward(out);
    out = c4->forward(out);
    out = d4->forward(out);
    out = e4->forward(out);
    out = maxpool->forward(out);
    out = a5->forward(out);
    out = b5->forward(out);
    out = avgpool->forward(out);
    out = out.view({out.size(0), -1});
    out = linear->forward(out);
    return out;
}
