// load model from model/adder.onnx
// verify the model with input data
// output the result
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <random>

int main() {
    // 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeModel");

    // 创建 ONNX Runtime 会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 加载 ONNX 模型
    const char* model_path = "../model/adder.onnx";
    Ort::Session session(env, model_path, session_options);

    // 获取模型输入输出信息
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    const char* output_name = output_name_ptr.get();

    // 获取输入的形状
    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = tensor_info.GetShape();

    // 生成两个随机数作为输入
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10000.0);
    std::vector<float> input_data = {static_cast<float>(dis(gen)), static_cast<float>(dis(gen))};

    // 创建输入张量
    std::vector<int64_t> input_dims = {1, 2}; // 假设输入形状为 [1, 2]
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_dims.data(), input_dims.size());

    // 运行模型
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

    // 获取输出数据
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    // 打印输出结果
    std::cout << "输入: " << input_data[0] << " + " << input_data[1] << std::endl;
    std::cout << "输出: " << output_data[0] << std::endl;

    return 0;
}