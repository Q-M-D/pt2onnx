# test model in model/adder.pth

import torch
import numpy as np
import os, sys
from py_train import Adder

def test_model():
    # load data
    data = np.load(os.path.dirname(__file__)+"/data/data.npy")
    x = data[:, :2]
    y = data[:, 2]

    # load model
    model = Adder()
    model.load(os.path.dirname(__file__)+"/model/adder.pth")

    # test model
    correct = 0
    for i in range(x.shape[0]):
        x_input = torch.tensor(x[i], dtype=torch.float32).view(1, 2)
        y_output = torch.tensor(y[i], dtype=torch.float32).view(1, 1)
        y_pred = model(x_input)
        # print(f"input: {x_input}, output: {y_output}, prediction: {y_pred}")
        if torch.abs(y_pred - y_output) < 0.5:
            correct += 1
    print(f"total: {x.shape[0]}, correct: {correct}, accuracy: {correct/x.shape[0]}")

def random_test():
    model = Adder()
    model.load(os.path.dirname(__file__)+"/model/adder.pth")
    
    correct = 0
    test_num = 1000
    for _ in range(test_num):
        x = np.random.randint(1, 10000, 2)
        y = x.sum()
        x_input = torch.tensor(x, dtype=torch.float32).view(1, 2)
        y_pred = model(x_input)
        print(f"input: {x_input}, output: {y}, prediction: {y_pred}")
        if torch.abs(y_pred - y) < 0.5:
            correct += 1
    print(f"total: {test_num}, correct: {correct}, accuracy: {correct/test_num}")
    
    # x_input = torch.tensor([1, 2], dtype=torch.float32).view(1, 2)
    # y_pred = model(x_input)
    # print(f"input: {x_input}, prediction: {y_pred}")


if __name__ == "__main__":
    # test_model()
    random_test()