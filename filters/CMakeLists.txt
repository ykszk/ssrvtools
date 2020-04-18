PROJECT (label_filters)
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

add_subdirectory(pybind11)
pybind11_add_module(_label_filters label_filters.cpp)
