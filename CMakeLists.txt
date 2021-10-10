cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(torch_cifar10 VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(DOWNLOAD_DATASETS "Automatically download required datasets at build-time." OFF)
option(CREATE_SCRIPTMODULES "Automatically create all required scriptmodule files at build-time (requires python3)." OFF)

set(PYTORCH_VERSION "1.9.1")

set(CMAKE_CXX_STANDARD 14)
#--------------------------------------------------------------------------------------------
# following line solve the issue of Can't link a project using cmake with OpenCV and LibTorch
# -------------------------------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch/share/cmake")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")

# add torch package
find_package(Torch ${PYTORCH_VERSION} EXACT QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch")

message(STATUS "Torch library status:")
message(STATUS "    version: ${Torch_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}") 

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17) # To use C++ <filesystem>

# use openmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")


set(requiredlibs ${OpenCV_LIBS})
list(APPEND requiredlibs ${TORCH_LIBRARIES})
#list(APPEND requiredlibs "png16")

# --- AlexNet torch_cifar10
add_executable(AlexNet_cifar10)
target_sources(AlexNet_cifar10 PRIVATE  AlexNet/main.cpp
    AlexNet/alexnet.cpp
    AlexNet/alexnet.h
    AlexNet/modelsimpl.h
    cifar10.cpp
    cifar10.h
)
target_link_libraries(AlexNet_cifar10 ${TORCH_LIBRARIES})

set_target_properties(AlexNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)
# -------------------------------------------------------------------

# --- DenseNet torch_cifar10
add_executable(DenseNet_cifar10)
target_sources(DenseNet_cifar10 PRIVATE  DenseNet/main.cpp
    DenseNet/densenet.cpp
    DenseNet/densenet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(DenseNet_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(DenseNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- DPN torch_cifar10
add_executable(DPN_cifar10)
target_sources(DPN_cifar10 PRIVATE  DPN/main.cpp
    DPN/dpn.cpp
    DPN/dpn.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(DPN_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(DPN_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- EfficientNet torch_cifar10
add_executable(EfficientNet_cifar10)
target_sources(EfficientNet_cifar10 PRIVATE  EfficientNet/main.cpp
    EfficientNet/efficientnet.cpp
    EfficientNet/efficientnet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(EfficientNet_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(EfficientNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- GoogleNet torch_cifar10
add_executable(GoogleNet_cifar10)
target_sources(GoogleNet_cifar10 PRIVATE  GoogleNet/main.cpp
    GoogleNet/googlenet.cpp
    GoogleNet/googlenet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(GoogleNet_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(GoogleNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- LeNet torch_cifar10
add_executable(LeNet_cifar10)
target_sources(LeNet_cifar10 PRIVATE  LeNet/main.cpp
    LeNet/nnet.cpp
    LeNet/nnet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(LeNet_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(LeNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- MobileNet torch_cifar10
add_executable(MobileNet_cifar10)
target_sources(MobileNet_cifar10 PRIVATE  MobileNet/main.cpp
    MobileNet/mobilenetv1.cpp
    MobileNet/mobilenetv1.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(MobileNet_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(MobileNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- PnasNet torch_cifar10
add_executable(PnasNet_cifar10)
target_sources(PnasNet_cifar10 PRIVATE  PnasNet/main.cpp
    PnasNet/pnasnet.cpp
    PnasNet/pnasnet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(PnasNet_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(PnasNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- PreactResNet torch_cifar10
add_executable(PreactResNet_cifar10)
target_sources(PreactResNet_cifar10 PRIVATE  PreactResNet/main.cpp
    PreactResNet/PreactResNet.cpp
    PreactResNet/PreactResNet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(PreactResNet_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(PreactResNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- RegNet torch_cifar10
add_executable(RegNet_cifar10)
target_sources(RegNet_cifar10 PRIVATE  RegNet/main.cpp
    RegNet/regnet.cpp
    RegNet/regnet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(RegNet_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(RegNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


# ---------------------------------------------------------------------
# --- ResNet torch_cifar10
add_executable(ResNet_cifar10)
target_sources(ResNet_cifar10 PRIVATE  ResNet/main.cpp
    ResNet/resnet.cpp
    ResNet/resnet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(ResNet_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(ResNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- VGG torch_cifar10
add_executable(VGG_cifar10)
target_sources(VGG_cifar10 PRIVATE  VGG/main.cpp
    VGG/vgg.cpp
    VGG/vgg.h
    VGG/modelsimpl.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(VGG_cifar10 ${TORCH_LIBRARIES})  

set_target_properties(VGG_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)
