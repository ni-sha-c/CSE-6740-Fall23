# implement support vector classifier
# Modified from https://github.com/kazuto1011/svm-pytorch/blob/master/main.py (Kazuto Nakashima) MIT License


import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset, DataLoader

def create_model():
    m = nn.Linear(2,1)
    m.to(args.device)
    return m

def create_data():
    X, Y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=2)
    Y[Y == 0] = -1
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    return X, Y

def create_dataset(X, Y):
    class dataset(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
        def __len__(self):
            return len(self.Y)
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    trainset = dataset(X,Y)
    return trainset
    

def train(X, Y, model, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    trainset = create_dataset(X, Y)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    N = len(trainset)
    model.train()
    for epoch in range(args.epoch):
        sum_loss = 0.0
        for t, (x, y) in enumerate(trainloader):
            x_g = x.to(args.device)
            y_g = y.to(args.device)
            output = model(x_g).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y_g * output, min=0))
            loss += args.c * (weight.t() @ weight) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))


def visualize(X, Y, model):
    W = model.weight.squeeze().detach().cpu().numpy()
    b = model.bias.squeeze().detach().cpu().numpy()

    delta = 0.001
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    x2 = -b / W[1] - W[0] / W[1] * x1
    x2_p = (1-b) / W[1] - W[0] / W[1] * x1 
    x2_m = (-1-b) / W[1] - W[0] / W[1] * x1
    
    x1_data = X[:, 0].detach().cpu().numpy()
    x2_data = X[:, 1].detach().cpu().numpy()
    y_data = Y.detach().cpu().numpy()

    x1_c1 = x1_data[np.where(y_data == 1)]
    x1_c2 = x1_data[np.where(y_data == -1)]
    x2_c1 = x2_data[np.where(y_data == 1)]
    x2_c2 = x2_data[np.where(y_data == -1)]

    plt.figure(figsize=(10, 10))
    plt.plot(x1_c1, x2_c1, "P", c="b", ms=10)
    plt.plot(x1_c2, x2_c2, "o", c="r", ms=10)
    plt.plot(x1, x2, "k", lw=3)
    plt.plot(x1, x2_p, "b--", lw=3)
    plt.plot(x1, x2_m, "r--", lw=3)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)

    X, Y = create_data()
    model = create_model()
    train(X, Y, model, args)
    visualize(X, Y, model)
