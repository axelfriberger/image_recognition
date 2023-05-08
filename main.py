import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import data_folder
import time
import saved_models

class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255.
        self.y = F.one_hot(self.y, num_classes=10).to(float)
    def __len__(self): 
        return self.x.shape[0]
    def __getitem__(self, ix): 
        return self.x[ix], self.y[ix]

train_ds = CTDataset("/Users/ag72273/MNIST/processed/training.pt")
test_ds = CTDataset("/Users/ag72273/MNIST/processed/test.pt")
"""
train_ds = CTDataset("C:\\Users\\Axel Friberger\\Downloads\\MNIST.tar\\MNIST\\MNIST\\processed\\training.pt")
test_ds = CTDataset("C:\\Users\\Axel Friberger\\Downloads\\MNIST.tar\\MNIST\\MNIST\\processed\\test.pt")
"""
train_dl = DataLoader(train_ds, batch_size=5)


class NeuralNet_1layer(nn.Module):
    def __init__(self, n_neurons, activation=nn.ReLU()):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, n_neurons)
        self.Matrix2 = nn.Linear(n_neurons, 10)
        self.R = activation

    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        return x.squeeze()


class NeuralNet_2layers(nn.Module):
    def __init__(self, n_neurons, activation=nn.ReLU()):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, n_neurons)
        self.Matrix2 = nn.Linear(n_neurons, n_neurons)
        self.Matrix3 = nn.Linear(n_neurons, 10)
        self.R = activation

    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()


class CNN(nn.Module):
    def __init__(self,n_neurons, activation=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7*7*32, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, 10)
        self.R = activation

    def forward(self, x):
        print(self.conv1.weight[0], "GAAAAAAAAAAP", self.conv2.weight[0][0      ])
        
        x = x.view(-1, 1, 28, 28)
        x = self.R(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.R(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*32)
        x = self.R(self.fc1(x))
        x = self.R(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()


def train_model(dl, f, n_epochs, L=nn.CrossEntropyLoss()):
    print(f.parameters())
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    
    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad() 
            loss_value = L(f(x.float()), y.float())
            loss_value.backward() 
            opt.step() 
            # Store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)
    

def plot_epoch_loss(epoch_data, loss_data):
    epoch_data_avgd = epoch_data.reshape(20,-1).mean(axis=1)
    loss_data_avgd = loss_data.reshape(20,-1).mean(axis=1)
    
    plt.plot(epoch_data_avgd, loss_data_avgd, "o--")
    plt.xlabel('Epoch Number')
    plt.ylabel('Cross Entropy')
    plt.title('Cross Entropy (per batch)')
    plt.show()


def accuracy(f, plot_misses=False):
    misses = []
    for x, y in test_ds:
        if f(x).argmax() != y.argmax():
            misses.append((x, y))

    if plot_misses:
        xs = [torch.Tensor(i) for i,j in misses[:40]] 
        ys = [j for i,j in misses[:40]] 

        yhats = [f(x).argmax() for x in xs]
        #yhats = f(xs).argmax()
        fig, ax = plt.subplots(10,4,figsize=(10,15))
        for i in range(40):
            plt.subplot(10,4,i+1)
            plt.imshow(xs[i])
            plt.title(f'Predicted Digit: {yhats[i]}')
        fig.tight_layout()
        plt.show()
    
    return(1-len(misses)/len(test_ds))


def showcase_model():
    f = NeuralNet_2layers(32, 32)
    epoch_data, loss_data = train_model(train_dl, f, n_epochs=10)
    plot_epoch_loss(epoch_data, loss_data)


def run_data(n_models, epochs, activation_function, step):
    with open("data_folder/data.txt", "a") as file:
        file.write(2*"\n")
        file.write(f"Two layers with (40 - {(n_models-1)*step+40}) neurons and {epochs} epochs with {activation_function} CE\n")

    for i in range(n_models):
        def run_function():
            if i == 0:
                f = NeuralNet_2layers(n_neurons=40)
            else:
                f = NeuralNet_2layers(n_neurons=step*(i)+40)

            train_model(train_dl, f, n_epochs=epochs)
            return accuracy(f)

        with open("data_folder/data.txt", "a") as file:
            if i==0:
                n = 40
            else:
                n = step*i+40
            start = time.time()
            acc = (run_function()+run_function())/2
            end = time.time()
            timer = end-start
            file.write(f"{n} {acc} {timer}\n")

def save_model(model, filename):
    torch.save(model.state_dict(), f"saved_models/{filename}")


def load_model(model_class, filepath, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(f"saved_models/{filepath}"))
    return model

if __name__ == "__main__":
    
    """f = CNN(n_neurons=50)
    train_model(train_dl, f, n_epochs=1)
    """
    x, y = train_ds[0]
    """fig, ax = plt.subplots(1,1,figsize=(6,9))
    for i in range(1):
        plt.subplot(1,1,i+1)
        plt.imshow(x)
        plt.title(f'Predicted Digit: {torch.argmax(y)}')
    fig.tight_layout()
    plt.show()"""
    
    #save_model(f, "CNN_50_neurons.pt")
    f = load_model(CNN, "CNN_50_neurons.pt", n_neurons=50)
    f(x)
