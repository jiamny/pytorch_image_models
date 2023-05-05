#include "resnet.h"
using Options = torch::nn::Conv2dOptions;

BasicBlock::BasicBlock(int64_t in_planes, int64_t planes, int64_t stride_) : shortcut(torch::nn::Sequential()) {

	  if( stride != stride_ ) stride = stride_;

	  conv1 = torch::nn::Conv2d(Options(in_planes, planes, 3).stride(stride).padding(1).bias(false));
	  bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	  conv2 = torch::nn::Conv2d(Options(planes, planes, 3).stride(1).padding(1).bias(false));

	  bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));


	  if( stride != 1 || in_planes != expansion*planes ) {
	  	   shortcut = torch::nn::Sequential(
	  	            		torch::nn::Conv2d(Options(in_planes, expansion*planes, 1).stride(stride).bias(false)),
	  						torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(expansion*planes)));
	  }

	  register_module("conv1", conv1);
	  register_module("bn1", bn1);
	  register_module("conv2", conv2);
	  register_module("bn2", bn2);

	  if(! shortcut->is_empty()) {
	        register_module("shortcut", shortcut);
	  }
}

torch::Tensor BasicBlock::forward(torch::Tensor x) {
	  at::Tensor residual(x.clone());

	  x = conv1->forward(x);
	  x = bn1->forward(x);
	  x = torch::relu(x);

	  x = conv2->forward(x);
	  x = bn2->forward(x);

	  if (! shortcut->is_empty()){
	       residual = shortcut->forward(residual);
	  }

	  x += residual;
	  x = torch::relu(x);

	  return x;
}

Bottleneck::Bottleneck(int64_t in_planes, int64_t planes, int64_t stride_) : shortcut(torch::nn::Sequential()) {

	  if( stride != stride_ ) stride = stride_;

	  conv1 = torch::nn::Conv2d(Options(in_planes, planes, 1).bias(false));

	  bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	  conv2 = torch::nn::Conv2d(Options(planes, planes, 3).stride(stride).padding(1).bias(false));
	  bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	  conv3 = torch::nn::Conv2d(Options(planes, expansion *planes,1).bias(false));
	  bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(expansion*planes));


	  if( stride != 1 || in_planes != expansion*planes ) {
		  shortcut = torch::nn::Sequential(
		    		torch::nn::Conv2d(Options(in_planes, expansion*planes, 1).stride(stride).bias(false)),
					torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(expansion*planes)));
	  }

	  register_module("conv1", conv1);
	  register_module("bn1", bn1);
	  register_module("conv2", conv2);
	  register_module("bn2", bn2);
	  register_module("conv3", conv3);
	  register_module("bn3", bn3);

	  if(! shortcut->is_empty()) {
		  register_module("shortcut", shortcut);
	  }
}

torch::Tensor Bottleneck::forward(torch::Tensor x) {
	  at::Tensor residual(x.clone());

	  x = conv1->forward(x);
	  x = bn1->forward(x).relu_();

	  x = conv2->forward(x);
	  x = bn2->forward(x).relu_();

	  x = conv3->forward(x);
	  x = bn3->forward(x);

	  if (! shortcut->is_empty()){
	       residual = shortcut->forward(residual);
	  }

	  x += residual;
	  x = torch::relu(x);

	  return x;
}

ResNetBB::ResNetBB(std::vector<int> num_blocks, int64_t num_classes) {
	    conv1 = torch::nn::Conv2d(Options(3, 64, 3).stride(1).padding(1).bias(false));
		bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));

		layer1 = _make_layer(64, num_blocks[0]);
		layer2 = _make_layer(128, num_blocks[1], 2);
		layer3 = _make_layer(256, num_blocks[2], 2);
		layer4 = _make_layer(512, num_blocks[3], 2);
		linear = torch::nn::Linear(512 * expansion, num_classes);

	    register_module("conv1", conv1);
	    register_module("bn1", bn1);
	    register_module("layer1", layer1);
	    register_module("layer2", layer2);
	    register_module("layer3", layer3);
	    register_module("layer4", layer4);
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
	      }
	    }
}

torch::Tensor ResNetBB::forward(torch::Tensor x) {
	  x = bn1->forward(conv1->forward(x)).relu_();

	  x = layer1->forward(x);
	  x = layer2->forward(x);
	  x = layer3->forward(x);
	  x = layer4->forward(x);

	  x = torch::nn::functional::avg_pool2d(x, torch::nn::functional::AvgPool2dFuncOptions(4));
	  x = linear->forward(x.view({x.size(0), -1}));

	  return x;
}

torch::nn::Sequential ResNetBB::_make_layer(
          int64_t planes,
          int64_t blocks,
          int64_t stride) {

		std::vector<int64_t> strides;
		strides.push_back(stride);

		for( int i = 0; i < (blocks-1); i++ ) //[1]*(num_blocks-1)
			strides.push_back(1);

		torch::nn::Sequential layers;

		for( int i = 0; i < strides.size(); i++ ) {
			layers->push_back(BasicBlock(in_planes, planes, strides[i]));
			this->in_planes = planes*expansion;
		}

		return layers;
}

ResNetBN::ResNetBN(std::vector<int> num_blocks, int64_t num_classes) {
	  conv1 = torch::nn::Conv2d(Options(3, 64, 3).stride(1).padding(1).bias(false));
	  bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));

	  layer1 = _make_layer(64, num_blocks[0], 1);
	  layer2 = _make_layer(128, num_blocks[1], 2);
	  layer3 = _make_layer(256, num_blocks[2], 2);
	  layer4 = _make_layer(512, num_blocks[3], 2);
	  linear = torch::nn::Linear(512*expansion, num_classes);

	  register_module("conv1", conv1);
	  register_module("bn1", bn1);
	  register_module("layer1", layer1);
	  register_module("layer2", layer2);
	  register_module("layer3", layer3);
	  register_module("layer4", layer4);
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
	     }
	  }
}

torch::Tensor ResNetBN::forward(torch::Tensor x) {
	  x = bn1->forward(conv1->forward(x)).relu_();

	  x = layer1->forward(x);
	  x = layer2->forward(x);
	  x = layer3->forward(x);
	  x = layer4->forward(x);

	  x = torch::nn::functional::avg_pool2d(x, torch::nn::functional::AvgPool2dFuncOptions(4));
	  x = linear->forward(x.view({x.size(0), -1}));

	  return x;
}

torch::nn::Sequential ResNetBN::_make_layer(
        int64_t planes,
        int64_t blocks,
        int64_t stride) {

		std::vector<int64_t> strides;
		strides.push_back(stride);

		for( int i = 0; i < (blocks-1); i++ ) //[1]*(num_blocks-1)
			strides.push_back(1);

		torch::nn::Sequential layers;

		for( int i = 0; i < strides.size(); i++ ) {
			layers->push_back(BasicBlock(in_planes, planes, strides[i]));
			this->in_planes = planes*expansion;
		}

		return layers;
}

ResNetBB ResNet18(int64_t num_classes) {
	std::vector<int> blocks = {2, 2, 2, 2};
    return ResNetBB(blocks, num_classes);
}

ResNetBB ResNet34(int64_t num_classes) {
	std::vector<int> blocks = {3, 4, 6, 3};
    return ResNetBB(blocks, num_classes);
}

ResNetBN ResNet50(int64_t num_classes) {
	std::vector<int> blocks = {3, 4, 6, 3};
    return ResNetBN(blocks, num_classes);
}

ResNetBN ResNet101(int64_t num_classes) {
	std::vector<int> blocks = {3, 4, 23, 3};
    return ResNetBN(blocks, num_classes);
}
