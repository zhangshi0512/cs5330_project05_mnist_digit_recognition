import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Define a pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # There are 10 classes in MNIST

    def forward(self, x):
        # Apply the first convolutional layer and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolutional layer and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        # Apply the first fully connected layer
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer
        x = self.fc2(x)
        return x
