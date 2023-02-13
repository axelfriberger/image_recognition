import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt

x = torch.tensor([[6, 2], [5, 2], [1, 3], [7, 6]]).float()
y = torch.tensor([1, 5 ,2, 5]).float()

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2, 8, bias = False)
        self.Matrix2 = nn.Linear(8, 1, bias = False)
    def forward(self, x):
        x = self.Matrix1(x)
        x = self.Matrix2(x)
        return x.squeeze()

f = MyNeuralNetwork()

yhat = f(x)

L = nn.MSELoss()

opt = SGD(f.parameters(), lr=0.001)

num_epochs = 50
losses = []
for _ in range(num_epochs):
    opt.zero_grad()
    loss_value = L(f(x), y)
    loss_value.backward()
    opt.step()
    losses.append(loss_value.item())

plt.plot(losses)
plt.ylabel('Loss $L(y,\hat{y};a)$')
plt.xlabel('Epochs')
plt.show()

print(f(x), "\n", y)