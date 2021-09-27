#include "resnet.h"

using Options = torch::nn::Conv2dOptions;

BasicBlockImpl::BasicBlockImpl(int64_t in_planes, int64_t planes, int64_t stride) {
	if( this->stride != stride ) this->stride = stride;

	this->conv1 = torch::nn::Conv2d(Options(in_planes, planes, 3).stride(this->stride).padding(1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	this->conv2 = torch::nn::Conv2d(Options(planes, planes, 3).stride(1).padding(1).bias(false));

	this->bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));


	if( this->stride != 1 || in_planes != this->expansion*planes ) {
	    this->shortcut = torch::nn::Sequential(
	            		torch::nn::Conv2d(Options(in_planes, this->expansion*planes, 1).stride(this->stride).bias(false)),
						torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(this->expansion*planes)));
	    useShortcut = true;
	}
}

torch::Tensor BasicBlockImpl::forward(torch::Tensor x) {

  auto out = conv1->forward(x);
  out = bn1->forward(out).relu_();

  out = conv2->forward(out);
  out = bn2->forward(out);

  if( this->useShortcut )
      out += this->shortcut->forward(x);
  else
	  out += x;

  return out.relu_();
}

BottleneckImpl::BottleneckImpl(int64_t in_planes, int64_t planes, int64_t stride) {

	if( this->stride != stride ) this->stride = stride;

	this->conv1 = torch::nn::Conv2d(Options(in_planes, planes, 1).bias(false));

	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	this->conv2 = torch::nn::Conv2d(Options(planes, planes, 3).stride(this->stride).padding(1).bias(false));
	this->bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	this->conv3 = torch::nn::Conv2d(Options(planes, this->expansion *planes,1).bias(false));
	this->bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(this->expansion*planes));


	if( this->stride != 1 || in_planes != this->expansion*planes ) {
	    this->shortcut = torch::nn::Sequential(
	    		torch::nn::Conv2d(Options(in_planes, this->expansion*planes, 1).stride(this->stride).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(this->expansion*planes)));
	    this->useShortcut = true;
	}
}

torch::Tensor BottleneckImpl::forward(torch::Tensor X) {

  auto out = conv1->forward(X);
  out = bn1->forward(out).relu_();

  out = conv2->forward(out);
  out = bn2->forward(out).relu_();

  out = conv3->forward(out);
  out = bn3->forward(out);

  if( this->useShortcut )
      out += this->shortcut->forward(X);
  else
	  out += X;

  return out.relu_();
}

ResNetBBImpl::ResNetBBImpl(std::vector<int> num_blocks, int64_t num_classes) {
	this->conv1 = torch::nn::Conv2d(Options(3, 64, 3).stride(1).padding(1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));

	this->layer1 = _make_layer(64, num_blocks[0], 1);
	this->layer2 = _make_layer(128, num_blocks[1], 2);
	this->layer3 = _make_layer(256, num_blocks[2], 2);
	this->layer4 = _make_layer(512, num_blocks[3], 2);
	this->linear = torch::nn::Linear(512*this->expansion, num_classes);
}

std::vector<BasicBlock> ResNetBBImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride) {
	std::vector<int64_t> strides;
	strides.push_back(stride);

	for( int i = 0; i < (blocks-1); i++ ) //[1]*(num_blocks-1)
		strides.push_back(1);

	std::vector<BasicBlock> layers;

	for( int i = 0; i < strides.size(); i++ ) {
		layers.push_back(BasicBlock(this->in_planes, planes, strides[i]));
		   this->in_planes = planes*this->expansion;
	}

	return layers;
}

torch::Tensor ResNetBBImpl::forward(torch::Tensor x) {
// out = self.conv1(x)
	auto out = bn1->forward(conv1->forward(x)).relu_();

//	std::cout << out.sizes() << std::endl;

	for( int i =0; i < layer1.size(); i++ ) {
			out = layer1[i]->forward(out);
//			std::cout << "layer1 - " << i << " >> " << out.sizes() << std::endl;
	}

	for( int i =0; i < layer2.size(); i++ )
		out = layer2[i]->forward(out);

	for( int i =0; i < layer3.size(); i++ )
		out = layer3[i]->forward(out);

	for( int i =0; i < layer4.size(); i++ )
		out = layer4[i]->forward(out);

	// out = F.avg_pool2d(out, 4)
    // out = out.view(out.size(0), -1)
    // out = self.linear(out)
	out = torch::nn::functional::avg_pool2d(out, torch::nn::functional::AvgPool2dFuncOptions(4));
	out = linear->forward(out.view({out.size(0), -1}));

	return out;
}

// ----------------------
ResNetBNImpl::ResNetBNImpl(std::vector<int> num_blocks, int64_t num_classes) {
	this->conv1 = torch::nn::Conv2d(Options(3, 64, 3).stride(1).padding(1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));

	this->layer1 = _make_layer(64, num_blocks[0], 1);
	this->layer2 = _make_layer(128, num_blocks[1], 2);
	this->layer3 = _make_layer(256, num_blocks[2], 2);
	this->layer4 = _make_layer(512, num_blocks[3], 2);
	this->linear = torch::nn::Linear(512*this->expansion, num_classes);
}

std::vector<Bottleneck> ResNetBNImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride) {
	std::vector<int64_t> strides;
	strides.push_back(stride);

	for( int i = 0; i < (blocks-1); i++ ) //[1]*(num_blocks-1)
		strides.push_back(1);


	std::vector<Bottleneck> layers;

	for( int i = 0; i < strides.size(); i++ ) {
		layers.push_back(Bottleneck(this->in_planes, planes, strides[i]));
		   this->in_planes = planes*this->expansion;
	}

	return layers;
}

torch::Tensor ResNetBNImpl::forward(torch::Tensor x) {
// out = self.conv1(x)
	auto out = bn1->forward(conv1->forward(x)).relu_();

//	std::cout << out.sizes() << std::endl;

	for( int i =0; i < layer1.size(); i++ ) {
			out = layer1[i]->forward(out);
//			std::cout << "layer1 - " << i << " >> " << out.sizes() << std::endl;
	}

	for( int i =0; i < layer2.size(); i++ )
		out = layer2[i]->forward(out);

	for( int i =0; i < layer3.size(); i++ )
		out = layer3[i]->forward(out);

	for( int i =0; i < layer4.size(); i++ )
		out = layer4[i]->forward(out);

	// out = F.avg_pool2d(out, 4)
    // out = out.view(out.size(0), -1)
    // out = self.linear(out)
	out = torch::nn::functional::avg_pool2d(out, torch::nn::functional::AvgPool2dFuncOptions(4));
	out = linear->forward(out.view({out.size(0), -1}));

	return out;
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

ResNetBN ResNet152(int64_t num_classes) {
	std::vector<int> blocks = {3, 8, 36, 3};
    return ResNetBN(blocks, num_classes);
}



