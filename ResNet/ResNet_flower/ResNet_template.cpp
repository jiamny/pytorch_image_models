#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../../transforms.hpp"              // transforms_Compose
#include "../../datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../../dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                      int64_t stride=1, int64_t padding=0, bool with_bias=false){
  torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes,
		  out_planes, kerner_size); //.stride(stride).padding(padding).bias(with_bias);
  conv_options.stride(stride);
  conv_options.padding(padding);
  conv_options.bias(with_bias);
  return conv_options;
}


struct BasicBlock : torch::nn::Module {

  static const int expansion;

  int64_t stride;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm2d bn2;
  torch::nn::Sequential downsample;

  BasicBlock(int64_t inplanes, int64_t planes, int64_t stride_=1,
             torch::nn::Sequential downsample_=torch::nn::Sequential())
      : conv1(conv_options(inplanes, planes, 3, stride_, 1)),
        bn1(planes),
        conv2(conv_options(planes, planes, 3, 1, 1)),
        bn2(planes),
        downsample(downsample_)
        {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    stride = stride_;
    if (!downsample->is_empty()){
      register_module("downsample", downsample);
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    at::Tensor residual(x.clone());
    
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);

    if (!downsample->is_empty()){
      residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
  }
};

const int BasicBlock::expansion = 1;


struct BottleNeck : torch::nn::Module {

  static const int expansion;

  int64_t stride;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm2d bn2;
  torch::nn::Conv2d conv3;
  torch::nn::BatchNorm2d bn3;
  torch::nn::Sequential downsample;

  BottleNeck(int64_t inplanes, int64_t planes, int64_t stride_=1,
             torch::nn::Sequential downsample_=torch::nn::Sequential())
      : conv1(conv_options(inplanes, planes, 1)),
        bn1(planes),
        conv2(conv_options(planes, planes, 3, stride_, 1)),
        bn2(planes),
        conv3(conv_options(planes, planes * expansion , 1)),
        bn3(planes * expansion),
        downsample(downsample_)
        {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    stride = stride_;
    if (!downsample->is_empty()){
      register_module("downsample", downsample);
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    at::Tensor residual(x.clone());

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);
    x = torch::relu(x);

    x = conv3->forward(x);
    x = bn3->forward(x);

    if (!downsample->is_empty()){
      residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
  }
};

const int BottleNeck::expansion = 4;


template <class Block> struct ResNet : torch::nn::Module {

  int64_t inplanes = 64;
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Sequential layer1{nullptr};
  torch::nn::Sequential layer2{nullptr};
  torch::nn::Sequential layer3{nullptr};
  torch::nn::Sequential layer4{nullptr};
  torch::nn::Linear fc{nullptr};

  ResNet(std::vector<int>  layers, int64_t num_classes=1000)
      //: conv1(conv_options(3, 64, 7, 2, 3)),
      //  bn1(64),
      //  layer1(_make_layer(64, layers[0])),
      //  layer2(_make_layer(128, layers[1], 2)),
      //  layer3(_make_layer(256, layers[2], 2)),
      //  layer4(_make_layer(512, layers[3], 2)),
      //  fc(512 * Block::expansion, num_classes)
        {

	this->conv1 = torch::nn::Conv2d(conv_options(3, 64, 7, 2, 3));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));

	this->layer1 = _make_layer(64, layers[0]);
	this->layer2 = _make_layer(128, layers[1], 2);
	this->layer3 = _make_layer(256, layers[2], 2);
	this->layer4 = _make_layer(512, layers[3], 2);
	this->fc = torch::nn::Linear(512 * Block::expansion, num_classes);

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);

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

  torch::Tensor forward(torch::Tensor x){

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = torch::avg_pool2d(x, 7, 1);
    x = x.view({x.sizes()[0], -1});
    x = fc->forward(x);

    return x;
  }


private:
  torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1){
    torch::nn::Sequential downsample;
    if (stride != 1 or inplanes != planes * Block::expansion){
      downsample = torch::nn::Sequential(
          torch::nn::Conv2d(conv_options(inplanes, planes * Block::expansion, 1, stride)),
          torch::nn::BatchNorm2d(planes * Block::expansion)
      );
    }
    torch::nn::Sequential layers;
    layers->push_back(Block(inplanes, planes, stride, downsample));
    inplanes = planes * Block::expansion;
    for (int64_t i = 0; i < blocks; i++){
      layers->push_back(Block(inplanes, planes));
    }

    return layers;
  }
};


ResNet<BasicBlock> resnet18( int64_t num_classes ){
  ResNet<BasicBlock> model({2, 2, 2, 2}, num_classes);
  return model;
}

ResNet<BasicBlock> resnet34( int64_t num_classes ){
  ResNet<BasicBlock> model({3, 4, 6, 3}, num_classes);
  return model;
}

ResNet<BottleNeck> resnet50( int64_t num_classes ){
  ResNet<BottleNeck> model({3, 4, 6, 3}, num_classes);
  return model;
}

ResNet<BottleNeck> resnet101( int64_t num_classes ){
  ResNet<BottleNeck> model({3, 4, 23, 3}, num_classes);
  return model;
}

ResNet<BottleNeck> resnet152( int64_t num_classes ){
  ResNet<BottleNeck> model({3, 8, 36, 3}, num_classes);
  return model;
}


int main() {

    torch::Device device("cpu");

    if (torch::cuda::is_available()){
    	device = torch::Device("cuda:0");
    }

    torch::Tensor t = torch::rand({2, 3, 224, 224}).to(device);
    ResNet<BottleNeck> resnet = resnet101( 5 );
    resnet.to(device);

    t = resnet.forward(t);
    std::cout << t.sizes()  << " " << t.device() << std::endl;
/*
	int64_t img_size = 224;
	size_t batch_size = 32;
	int64_t class_num = 5;
	const size_t valid_batch_size = 1;
	std::vector<std::string> class_names = {"daisy", "dandelion", "roses", "sunflowers", "tulips"};

	constexpr bool train_shuffle = true;    // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;  	// the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;    // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 2;     // the number of workers to retrieve data from the validation dataset

    bool valid = true;						// has valid dataset
    bool test  = true;						// has test dataset

    // (4) Set Transforms
    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                     // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "/media/stree/localssd/DL_data/flower_data2/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset, test_dataset;      		// dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader, test_dataloader; 	// dataloader;

    // (1) Get Dataset
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, train_shuffle, train_workers);

	std::cout << "total training images : " << dataset.size() << std::endl;

    std::string valid_dataroot = "/media/stree/localssd/DL_data/flower_data2/val";
    valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
    valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, valid_batch_size, valid_shuffle, valid_workers);

    std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    bool vobose = false;

    ResNet<BasicBlock> model = resnet18(class_num);
	model.to(device);
	std::cout << model << std::endl;

	auto dict = model.named_parameters();
	for (auto n = dict.begin(); n != dict.end(); n++) {
		std::cout << (*n).key() << std::endl;
	}

	std::cout << "Test model ..." << std::endl;
	torch::Tensor x = torch::randn({1,3, img_size, img_size}).to(device);
	torch::Tensor y = model.forward(x);
	std::cout << y << std::endl;

	torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));

	auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

	size_t epoch;
	size_t total_iter = dataloader.get_count_max();
	size_t start_epoch, total_epoch;
	start_epoch = 1;
	total_iter = dataloader.get_count_max();
	total_epoch = 45;

	bool first = true;
	std::vector<float> train_loss_ave;
	std::vector<float> train_epochs;

	for (epoch = start_epoch; epoch <= total_epoch; epoch++) {
		model.train();
		std::cout << "--------------- Training --------------------\n";
		first = true;
		float loss_sum = 0.0;
		while (dataloader(mini_batch)) {
			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);

			if( first && vobose ) {
				for(size_t i = 0; i < label.size(0); i++)
					std::cout << label[i].item<int64_t>() << " ";
				std::cout << "\n";
				first = false;
			}

			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);
			output = model.forward(image);
			auto out = torch::nn::functional::log_softmax(output, 1); // dim
			//std::cout << output.sizes() << "\n" << out.sizes() << std::endl;

			loss = criterion(out, label); //torch::mse_loss(out, label);

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();

			loss_sum += loss.item<float>();
		}

		train_loss_ave.push_back(loss_sum/total_iter);
		train_epochs.push_back(epoch*1.0);
		std::cout << "epoch: " << epoch << "/"  << total_epoch << ", avg_loss: " << (loss_sum/total_iter) << std::endl;

		// ---------------------------------
		// validation
		// ---------------------------------
		if( valid && (epoch % 1 == 0) ) {
			std::cout << "--------------- validation ------------------\n";
			model.eval();
			torch::NoGradGuard no_grad;

			size_t iteration = 0;
			float total_loss = 0.0;
			size_t total_match = 0, total_counter = 0;
			torch::Tensor responses;
			first = true;
			while (valid_dataloader(mini_batch)){

				image = std::get<0>(mini_batch).to(device);
			    label = std::get<1>(mini_batch).to(device);
			    size_t mini_batch_size = image.size(0);

			    if( first && vobose ) {
			    	for(size_t i = 0; i < label.size(0); i++)
			    		std::cout << label[i].item<int64_t>() << " ";
			    	std::cout << "\n";
			    	first = false;
			    }

			    output = model.forward(image);
			    auto out = torch::nn::functional::log_softmax(output, 1); // dim=
			    loss = criterion(out, label);

			    responses = output.exp().argmax(1); // dim

			    for (size_t i = 0; i < mini_batch_size; i++){
			        int64_t response = responses[i].item<int64_t>();
			        int64_t answer = label[i].item<int64_t>();

			        total_counter++;
			        if (response == answer) total_match++;
			    }
			    total_loss += loss.item<float>();
			    iteration++;
			}
			// (3) Calculate Average Loss
			float ave_loss = total_loss / (float)iteration;

			// (4) Calculate Accuracy
			float total_accuracy = (float)total_match / (float)total_counter;
			std::cout << "Validation accuracy: " << total_accuracy << std::endl << std::endl;
		}
	}

	//
	if( test ) {
		std::cout << "--------------- Testing ---------------------\n";
		std::string test_dataroot = "/media/stree/localssd/DL_data/flower_data2/val";
		test_dataset = datasets::ImageFolderClassesWithPaths(test_dataroot, transform, class_names);

		test_dataloader = DataLoader::ImageFolderClassesWithPaths(test_dataset, 1, false, 0);

		std::cout << "total test images : " << test_dataset.size() << std::endl << std::endl;

		float  ave_loss = 0.0;
		size_t match = 0;
		size_t counter = 0;
		std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> data;
		std::vector<size_t> class_match = std::vector<size_t>(class_num, 0);
		std::vector<size_t> class_counter = std::vector<size_t>(class_num, 0);
		std::vector<float> class_accuracy = std::vector<float>(class_num, 0.0);

	    model.eval();
	    torch::NoGradGuard no_grad;

	    while( test_dataloader(data) ){
	        image = std::get<0>(data).to(device);
	        label = std::get<1>(data).to(device);
	        output = model.forward(image);
	        auto out = torch::nn::functional::log_softmax(output, 1);

	        loss = criterion(out, label);

	        ave_loss += loss.item<float>();

	        output = output.exp();

	        int64_t response = output.argmax(1).item<int64_t>();

	        int64_t answer = label[0].item<int64_t>();
	        counter += 1;
	        class_counter[answer]++;

	        if (response == answer){
	        	class_match[answer]++;
	            match += 1;
	        }
	    }

	    // (7.1) Calculate Average
	    ave_loss = ave_loss / (float)dataset.size();

	    // (7.2) Calculate Accuracy
	    std::cout << "Test accuracy ==========\n";
	    for (size_t i = 0; i < class_num; i++){
	    	class_accuracy[i] = (float)class_match[i] / (float)class_counter[i];
	    	std::cout << class_names[i] << ": " << class_accuracy[i] << "\n";
	    }
	    float accuracy = (float)match / float(counter);
	    std::cout << "\nTest accuracy: " << accuracy << std::endl;
	}
*/
    std::cout << "Done!\n";

}
