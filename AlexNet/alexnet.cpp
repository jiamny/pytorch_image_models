#include "alexnet.h"

#include "modelsimpl.h"

AlexNetImpl::AlexNetImpl(int64_t num_classes) {
  features = torch::nn::Sequential(
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(3, 64, 3).stride(2).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::max_pool2d, 2, 2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::max_pool2d, 2, 2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::max_pool2d, 2, 2));

  classifier = torch::nn::Sequential(
      torch::nn::Dropout(torch::nn::Dropout2dOptions().p(0.5)),
      torch::nn::Linear(256 * 6 * 6, 4096),
      torch::nn::Functional(torch::relu),
      torch::nn::Dropout(torch::nn::Dropout2dOptions().p(0.5)),
      torch::nn::Linear(4096, 4096),
      torch::nn::Functional(torch::relu),
	  torch::nn::Linear(4096, num_classes));

  register_module("features", features);
  register_module("classifier", classifier);
}

torch::Tensor AlexNetImpl::forward(torch::Tensor x) {
  x = features->forward(x);
  x = torch::adaptive_avg_pool2d(x, {6, 6});
  x = x.view({x.size(0), -1});
  x = classifier->forward(x);

  return x;
}

