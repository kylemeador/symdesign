cmake_minimum_required(VERSION 3.4...3.18)
project(load_armadillo_to_numpy LANGUAGES CXX)
find_package(pybind11 CONFIG)

ADD_SUBDIRECTORY(carma)
#include_directories(${pybind11_DIR}) # pybind11)
#find_package(pybind11 REQUIRED)
# add_subdirectory(${pybind11_DIR}) # pybind11) must be a subdirectory for this
# find_package(${Python3_EXECUTABLE})
pybind11_add_module(load_armadillo_to_numpy load_armadillo_to_numpy.cpp)
TARGET_LINK_LIBRARIES(load_armadillo_to_numpy PRIVATE carma)


#project(example LANGUAGES CXX)

## include_directories(${pybind11_DIR})
#find_package(pybind11 REQUIRED)
#pybind11_add_module(example example.cpp)