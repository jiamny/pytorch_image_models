cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

find_package(OpenMP REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIRS})

set(requiredlibs "python3.10")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")

list(APPEND requiredlibs ${PNG_LIBRARIES})
list(APPEND requiredlibs ${OpenMP_CXX_LIBRARIES})

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()


# ---------------------------------------------------------------------
# --- ResNet torch_cifar10
add_executable(ResNet_cifar10)
target_sources(ResNet_cifar10 PRIVATE  ResNet_32/main.cpp
    ResNet_32/resnet.cpp
    ResNet_32/resnet.h
    ../cifar10.cpp
    ../cifar10.h
)

#target_link_libraries(ResNet_cifar10 ${TORCH_LIBRARIES})  
target_link_libraries( ResNet_cifar10 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )  

set_target_properties(ResNet_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- ResNet torch_flower_photos
add_executable(ResNet_flower_photos)
target_sources(ResNet_flower_photos PRIVATE  ResNet_flower/ResNet_template.cpp
    ../dataloader.hpp
	../dataloader.cpp
	../datasets.hpp
	../datasets.cpp
	../transforms.hpp
	../transforms.cpp
)

target_link_libraries( ResNet_flower_photos ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )  

set_target_properties(ResNet_flower_photos PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)



