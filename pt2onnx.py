# Tranns pth to onnx:

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import torch.onnx
from py_train import Adder

if __name__ == "__main__":
    model = Adder()
    model.load(os.path.dirname(__file__)+"/model/adder.pth")
    model.eval()
    x_input = torch.tensor([1, 2], dtype=torch.float32).view(1, 2)
    y_pred = model(x_input)
    print(f"input: {x_input}, prediction: {y_pred}")
    torch.onnx.dynamo_export(model, x_input).save(os.path.dirname(
        __file__)+"/model/adder.onnx")
