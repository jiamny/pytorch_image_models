
#include "efficientnet.h"
#include <torch/torch.h>

using Options = torch::nn::Conv2dOptions;

torch::Tensor swish(torch::Tensor x) {
	//x * x.sigmoid()
	return (x * x.sigmoid());
}

torch::Tensor drop_connect(torch::Tensor x, double drop_ratio) {
	auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
	double keep_ratio = 1.0 - drop_ratio;
	torch::Tensor mask = torch::empty({x.sizes()[0], 1, 1, 1}, options); //, x.dtype, x.device);
	mask.bernoulli_(keep_ratio);
	x.div_(keep_ratio);
	x.mul_(mask);
	return x;
}

SEImpl::SEImpl(int64_t in_planes, int64_t se_planes, torch::Device device) {
	se1 = torch::nn::Conv2d(Options(in_planes, se_planes, 1).bias(true));
	se2 = torch::nn::Conv2d(Options(se_planes, in_planes, 1).bias(true));
	se1->to(device);
	se2->to(device);
	register_module("se1", se1);
	register_module("se2", se2);
}

torch::Tensor SEImpl::forward(torch::Tensor x) {
	at::Tensor kp(x.clone());
	//kp = kp.to(x.device());

    x = torch::adaptive_avg_pool2d(x, {1, 1}).to(x.device());
    //se1->to(x.device());

    x = swish(se1->forward(x));

    //se2->to(x.device());
    x = se2->forward(x).sigmoid();

    x = kp * x;
    return x;
}

Block_Impl::Block_Impl( int64_t in_planes,
    int64_t out_planes,
    int64_t kernel_size,
    int64_t stride_,
    int64_t expand_ratio_,
    double se_ratio_,
    double drop_rate_, torch::Device device) {

	stride = stride_;
	expand_ratio = expand_ratio_;
	se_ratio = se_ratio_;
	drop_rate = drop_rate_;

	//Expansion
	int64_t planes = expand_ratio * in_planes;

	conv1 = torch::nn::Conv2d(Options(in_planes, planes, 1).stride(1).padding(0).bias(false));
	conv1->to(device);
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));
	bn1->to(device);

	//Depthwise conv
	conv2 = torch::nn::Conv2d(Options(planes, planes, kernel_size).stride(stride).padding( (kernel_size == 3) ?
										1 : 2).groups(planes).bias(false));
	conv2->to(device);
	bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));
	bn2->to(device);

	//SE layers
	int64_t se_planes = static_cast<int64_t>(in_planes * se_ratio);
	se = SE(planes, se_planes, device);

	//Output
	conv3 = torch::nn::Conv2d(Options(planes, out_planes, 1).stride(1).padding(0).bias(false));
	bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
	conv3->to(device);
	bn3->to(device);

	// Skip connection if in and out shapes are the same (MV-V2 style)
	has_skip = ((stride == 1) && (in_planes == out_planes));

	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("conv2", conv2);
	register_module("bn2", bn2);
	register_module("conv3", conv3);
	register_module("bn3", bn3);

}

torch::Tensor Block_Impl::forward(torch::Tensor x) {
	at::Tensor kp(x.clone());
	//kp = kp.to(x.device());

    x = (expand_ratio == 1 ) ? x : swish(bn1->forward(conv1->forward(x)));

	x = swish(bn2->forward(conv2->forward(x)));

	x = se->forward(x);

	x = bn3->forward(conv3->forward(x));

	if( has_skip ) {
		if( training &&  drop_rate > 0 )
			x = drop_connect(x, drop_rate);
		x = x + kp; // gives you a new tensor with summation ran over out and x == out.add(x)
	}

	return x;
}

EfficientNetImpl::EfficientNetImpl(std::map<std::string, std::vector<int64_t>> cfg_, int64_t num_classes, torch::Device device) {
	cfg = cfg_;
	std::vector<int64_t> out_planes = cfg.at("out_planes");
	adavgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1));
	conv1 = torch::nn::Conv2d(Options(3, 32, 3).stride(1).padding(1).bias(false));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
	layers = _make_layers(32, device);
	linear = torch::nn::Linear(out_planes[out_planes.size()-1], num_classes);
	adavgpool->to(device);
	conv1->to(device);
	bn1->to(device);
	linear->to(device);

	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("layers", layers);
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

torch::nn::Sequential  EfficientNetImpl::_make_layers(int64_t in_planes, torch::Device device) {
	torch::nn::Sequential  layers;
	std::vector<int64_t> expansion = cfg.at("expansion");
	std::vector<int64_t> out_planes = cfg.at("out_planes");
	std::vector<int64_t> num_blocks = cfg.at("num_blocks");
	std::vector<int64_t> kernel_size = cfg.at("kernel_size");
	std::vector<int64_t> stride = cfg.at("stride");


	for( int j = 0; j < expansion.size(); j++ ) {
		std::vector<int64_t> strides;
		strides.push_back(stride[j]);

		for( int i = 0; i < (num_blocks[j]-1); i++ ) strides.push_back(1);

		for( int s = 0; s < strides.size(); s++ ) {
			layers->push_back(Block_(in_planes,
                    out_planes[j],
                    kernel_size[j],
                    strides[s],
                    expansion[j],
                    0.25,
                    0, device));
			in_planes = out_planes[j];
		}
	}
	return layers;
}

torch::Tensor EfficientNetImpl::forward(torch::Tensor x) {
	x = swish(bn1->forward(conv1->forward(x)));
    x = layers->forward(x);
//	for( int i =0; i < layers.size(); i++ ) {
//		out = layers[i]->forward(out);
//		std::cout << "layer " << i << " -> " << out.sizes() << std::endl;
//	}
    //out =  torch::nn::functional::adaptive_avg_pool2d(out, torch::nn::AdaptiveAvgPool2dOptions(1));
	x = adavgpool->forward(x);
//    std::cout << "pool2d -> " << out.sizes() << std::endl;
    x = x.view({x.size(0), -1});
    x = linear->forward(x);

    return x;
}

EfficientNet EfficientNetB0(int64_t num_classes, torch::Device device) {
	std::map<std::string, std::vector<int64_t>>  cfg;
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("num_blocks", {1, 2, 2, 3, 3, 4, 1}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("expansion", {1, 6, 6, 6, 6, 6, 6}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("out_planes", {16, 24, 40, 80, 112, 192, 320}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("kernel_size", {3, 3, 5, 3, 5, 5, 3}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("stride", {1, 2, 2, 2, 1, 2, 1}));
	return EfficientNet(cfg, num_classes, device);
}
