cmake_minimum_required(VERSION 3.10)
project(ONNXInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(ONNXRUNTIME_DIR "/usr/local")

include_directories(${ONNXRUNTIME_DIR}/include)

link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(onnx onnx.cpp)

target_link_libraries(onnx onnxruntime)