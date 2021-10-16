#include "resnext.h"

using Options = torch::nn::Conv2dOptions;

ResNextBlockImpl::ResNextBlockImpl(int64_t in_planes, int64_t cardinality, int64_t bottleneck_width, int64_t stride) {
	if( this->stride != stride ) this->stride = stride;
	if( this->cardinality != cardinality ) this->cardinality = cardinality;
	if( this->bottleneck_width != bottleneck_width ) this->bottleneck_width = bottleneck_width;

	int64_t group_width = cardinality * bottleneck_width;

	this->conv1 = torch::nn::Conv2d(Options(in_planes, group_width, 1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(group_width));
	this->conv2 = torch::nn::Conv2d(Options(group_width, group_width, 3).stride(stride).padding(1).groups(cardinality).bias(false));
	this->bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(group_width));
	this->conv3 = torch::nn::Conv2d(Options(group_width, expansion*group_width, 1).bias(false));
	this->bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(expansion*group_width));


	if( this->stride != 1 || in_planes != this->expansion*group_width ) {

		this->shortcut = torch::nn::Sequential(
				torch::nn::Conv2d(Options(in_planes, expansion*group_width, 1).stride(stride).bias(false)),
				torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(expansion*group_width))
		);
	    this->useShortcut = true;
	}
}

torch::Tensor ResNextBlockImpl::forward(torch::Tensor x) {

	auto out = torch::relu(bn1->forward(conv1->forward(x)));
	out = torch::relu(bn2->forward(conv2->forward(out)));
	out = bn3->forward(conv3->forward(out));

	if( this->useShortcut )
		out += this->shortcut->forward(x);
	else
		out += x;

  return out.relu_();
}

ResNextImpl::ResNextImpl(std::vector<int> num_blocks, int64_t cardinality, int64_t bottleneck_width, int64_t num_classes) {
	this->num_blocks = num_blocks;
	this->cardinality = cardinality;
	this->bottleneck_width = bottleneck_width;
	this->num_classes = num_classes;

	this->conv1 =torch::nn::Conv2d(Options(3, 64, 1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
	this->layer1 = _make_layer(num_blocks[0], 1);
	this->layer2 = _make_layer(num_blocks[1], 2);
	this->layer3 = _make_layer(num_blocks[2], 2);

	linear = torch::nn::Linear(cardinality*bottleneck_width*8, num_classes);
}

std::vector<ResNextBlock> ResNextImpl::_make_layer(int64_t blocks, int64_t stride) {
	std::vector<int64_t> strides;
	strides.push_back(stride);

	for( int i = 0; i < (blocks-1); i++ ) //[1]*(num_blocks-1)
		strides.push_back(1);

	std::vector<ResNextBlock> layers;

	for( int i = 0; i < strides.size(); i++ ) {
		layers.push_back(ResNextBlock(this->in_planes, this->cardinality, this->bottleneck_width, strides[i]));
		this->in_planes = this->expansion * this->cardinality * this->bottleneck_width;
	}
	this->bottleneck_width *= 2;

	return layers;
}

torch::Tensor ResNextImpl::forward(torch::Tensor x) {
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

	out = torch::nn::functional::avg_pool2d(out, torch::nn::functional::AvgPool2dFuncOptions(8));
	out = linear->forward(out.view({out.size(0), -1}));

	return out;
}


ResNext ResNeXt29_2x64d(int64_t num_classes) {
	std::vector<int> blocks = {3, 3, 3};
    return ResNext(blocks, 2, 64, num_classes);
}

ResNext ResNeXt29_4x64d(int64_t num_classes) {
	std::vector<int> blocks = {3, 3, 3};
    return ResNext(blocks, 4, 64, num_classes);
}

ResNext ResNeXt29_8x64d(int64_t num_classes){
	std::vector<int> blocks = {3, 3, 3};
    return ResNext(blocks, 8, 64, num_classes);
}

ResNext ResNeXt29_32x4d(int64_t num_classes) {
	std::vector<int> blocks = {3, 3, 3};
    return ResNext(blocks, 32, 4, num_classes);
}




