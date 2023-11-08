# Shi Zhang
# main.py is the entry point of the project that calls functions from other files

import torch
from network import MyNetwork
from train import train, save_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

def main():
    # Check if CUDA is available and set device to GPU if it is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MNIST Dataset transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize the network
    model = MyNetwork().to(device)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # Training loop
    for epoch in range(1, 11):  # 10 epochs
        train(model, device, train_loader, optimizer, epoch)
    
    # Save the trained model
    save_model(model)

if __name__ == '__main__':
    main()
