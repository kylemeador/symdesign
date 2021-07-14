cmake_minimum_required(VERSION 3.4...3.18)
project(bmdca LANGUAGES CXX)
find_package(pybind11 CONFIG)

ADD_SUBDIRECTORY(carma)
#include_directories(${pybind11_DIR}) # pybind11)
#find_package(pybind11 REQUIRED)
# add_subdirectory(${pybind11_DIR}) # pybind11) must be a subdirectory for this
# find_package(${Python3_EXECUTABLE})
pybind11_add_module(bmdca bmdca.cpp)
TARGET_LINK_LIBRARIES(bmdca PRIVATE carma)
