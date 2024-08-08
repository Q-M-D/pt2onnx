# train a model to add two numbers
# input: two numbers
# output: the sum of the two numbers
# model: a simple neural network with two dense layers
# data: 10000 pairs of numbers, each pair has three numbers, the first two are inputs and the last one is output
# model will be saved in model/adder.pth
# data is saved in data/data.npy
# model will be trained for 10 epochs

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys

class Adder(nn.Module):
    def __init__(self):
        super(Adder, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    # load data
    data = np.load(os.path.dirname(__file__)+"/data/data.npy")
    x = data[:, :2]
    y = data[:, 2]

    # create model
    model = Adder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # train model
    for epoch in range(20):
        for i in range(x.shape[0]):
            x_input = torch.tensor(x[i], dtype=torch.float32).view(1, 2)
            y_output = torch.tensor(y[i], dtype=torch.float32).view(1, 1)
            optimizer.zero_grad()
            y_pred = model(x_input)
            loss = criterion(y_pred, y_output)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1}, loss {loss.item()}")

    # save model
    if os.path.exists(os.path.dirname(__file__)+"/model") == False:
        os.makedirs(os.path.dirname(__file__)+"/model")
    model.save(os.path.dirname(__file__)+"/model/adder.pth")