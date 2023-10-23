# implement fully connected neural network with pytorch
# Mostly written by chatGPT
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Define a simple feedforward neural network
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (if not already flattened)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_model():
    input_size = 784
    hidden_size = 16
    output_size = 10
    model = FeedForwardNet(input_size, hidden_size, output_size)
    return model
   

def train(model, args):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # ADAM, ADAMW
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epoch):
        sum_loss = 0.0
        for t, (x, y) in enumerate(train_loader):
            
            output = model(x)
            #weight = model.weight.squeeze()

            
            loss = criterion(output, y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print("Epoch: {:4d}\tloss: {}".format(epoch, loss.item()))

    # Test model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print("Accuracy (%): {}".format(correct / total*100))
    return model, train_dataset, test_dataset

def visualize_model(model):
    weight = model.weight.squeeze()
    weight = weight.detach().numpy()
    weight = weight.reshape(10, 28, 28)
    weight = np.concatenate(weight, axis=1)
    plt.imshow(weight, cmap="gray")
    plt.show()


def visualize_dataset(dataset):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(dataset[i][0].squeeze(), cmap="gray")
        plt.title(str(dataset[i][1]))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)

    
    model = create_model()
    model, train_dataset, test_dataset = train(model, args)
   
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #for x, y in test_loader:
        #output = model(x)
        #_, predicted = torch.max(output.data, 1)
        #print("Predicted: {}".format(predicted.item()))
        #plt.imshow(x.squeeze().detach().numpy(), cmap="gray")
        #plt.show()
        #break
