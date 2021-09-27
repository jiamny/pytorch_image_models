#include "densenet.h"
#include <cmath>

using Options = torch::nn::Conv2dOptions;


BottleneckImpl::BottleneckImpl(int64_t in_planes, int64_t growth_rate) {
	 this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));
	 this->conv1 = torch::nn::Conv2d(Options(in_planes, 4*growth_rate, 1).bias(false));
	 this->bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(4*growth_rate));
	 this->conv2 = torch::nn::Conv2d(Options(4*growth_rate, growth_rate, 3).padding(1).bias(false));
}


torch::Tensor BottleneckImpl::forward(torch::Tensor x) {

	auto out = conv1->forward(torch::relu(bn1->forward(x)));
	out = conv2->forward(torch::relu(bn2->forward(out)));
	out = torch::cat({out,x}, 1);
	return out;
}


TransitionImpl::TransitionImpl(int64_t in_planes, int64_t out_planes){
	this->bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));
	this->conv = torch::nn::Conv2d(Options(in_planes, out_planes, 1).bias(false));
}

torch::Tensor TransitionImpl::forward(torch::Tensor x){
	auto  out = conv->forward(torch::relu(bn->forward(x)));
    out = torch::avg_pool2d(out, 2);
	return out;
}

// growth_rate=12, reduction=0.5
DenseNetImpl::DenseNetImpl(std::vector<int64_t> nblocks, int64_t growth_rate, double reduction, int64_t num_classes) {

	this->growth_rate = growth_rate;

	int64_t num_planes = 2*growth_rate;
	this->conv1 = torch::nn::Conv2d(Options(3, num_planes, 3).padding(1).bias(false));

	this->dense1 = _make_dense_layers(num_planes, nblocks[0]);

	num_planes += nblocks[0]*growth_rate;
	int64_t out_planes = static_cast<int64_t>(std::floor(num_planes*reduction));

	this->trans1 = Transition(num_planes, out_planes);
	num_planes = out_planes;

	this->dense2 = _make_dense_layers(num_planes, nblocks[1]);
	num_planes += nblocks[1]*growth_rate;
	out_planes = static_cast<int64_t>(std::floor(num_planes*reduction));
	this->trans2 = Transition(num_planes, out_planes);
	num_planes = out_planes;

	this->dense3 = _make_dense_layers(num_planes, nblocks[2]);
	num_planes += nblocks[2]*growth_rate;
	out_planes = static_cast<int64_t>(std::floor(num_planes*reduction));
	this->trans3 = Transition(num_planes, out_planes);
	num_planes = out_planes;

	this->dense4 = _make_dense_layers(num_planes, nblocks[3]);
	num_planes += nblocks[3]*growth_rate;

	this->bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_planes));
	this->linear = torch::nn::Linear(num_planes, num_classes);
}

torch::Tensor DenseNetImpl::forward(torch::Tensor x) {
    auto out = conv1->forward(x);

    for( size_t i = 0; i < dense1.size(); i++ ) {
    	out = dense1[i]->forward(out);
    	//std::cout << "se1 - " << i << " " << out.sizes() << std::endl;
    }

    out = trans1->forward(out);

    //out = self.trans1(self.dense1(out))
    for( size_t i = 0; i < dense2.size(); i++ )
    	out = dense2[i]->forward(out);

    out = trans2->forward(out);

    // out = self.trans2(self.dense2(out))
    for( size_t i = 0; i < dense3.size(); i++ )
    	out = dense3[i]->forward(out);

    out = trans3->forward(out);

    // out = self.trans3(self.dense3(out))
    for( size_t i = 0; i < dense4.size(); i++ )
    	out = dense4[i]->forward(out);

    // out = self.dense4(out)
     out = torch::avg_pool2d(torch::relu(bn->forward(out)), 4);
     out = out.view({out.size(0), -1});
     out = linear->forward(out);
     return out;
}

std::vector<Bottleneck> DenseNetImpl::_make_dense_layers(int64_t in_planes, int64_t nblock) {
	std::vector<Bottleneck> layers;

	for(int i = 0; i < nblock; i++ ) {
	    layers.push_back(Bottleneck(in_planes, this->growth_rate));
	    in_planes += this->growth_rate;
	}
	return layers;
}

