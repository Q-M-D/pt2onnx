# Python pth $\rightarrow$ C++ ONNX demo

以加法器为例，本项目展示了一个从Python pth模型训练生成到C++ ONNX模型文件加载和推理的完整流程。

## 1. Python pth模型训练生成

### 生成数据
```bash
python py_generater.py
```

### 训练并保存模型
```bash
python py_train.py
```

### 测试模型
```bash
python py_test.py
```

## 2. 转化C++ ONNX模型文件

### 转化模型
```bash
python py2onnx.py
```

## 3. C++ ONNX模型文件加载与使用

### 编译
```bash
mkdir build
cd build
cmake ..
make
```

### 运行
```bash
./onnx
```
