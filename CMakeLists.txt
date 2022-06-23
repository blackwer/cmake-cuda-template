cmake_minimum_required(VERSION 3.18)

project(
  cuda-template
  LANGUAGES CXX CUDA)

add_executable(main src/main.cu)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES "70;75")
