import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import OrderedDict

# Define the Gabor filter bank function
def generate_gabor_filters(size, num_filters):
    filters = []
    for i in range(num_filters):
        theta = np.pi * i / num_filters
        kernel = cv2.getGaborKernel((size, size), sigma=3.0, theta=theta, lambd=10.0, gamma=0.5, psi=0)
        filters.append(kernel)
    return torch.tensor(filters, dtype=torch.float32).unsqueeze(1)

# Define the new network with Gabor filters as the first layer
class GaborNetwork(nn.Module):
    def __init__(self, gabor_filters, num_fc_nodes):
        super(GaborNetwork, self).__init__()
        self.gabor_filters = nn.Parameter(gabor_filters, requires_grad=False)
        # Define the rest of the network architecture
        self.conv2 = nn.Conv2d(gabor_filters.shape[0], 20, kernel_size=5)
        # Assuming the output of conv2 is 20*4*4 after pooling and convolutions
        self.fc1 = nn.Linear(20*4*4, num_fc_nodes)
        self.fc2 = nn.Linear(num_fc_nodes, 10)  # MNIST has 10 classes

    def forward(self, x):
        x = F.conv2d(x, self.gabor_filters, padding=2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 20*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load the MNIST dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True)

# Define training and evaluation functions
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_accuracy(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

# Initialize the model with Gabor filters and train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gabor_filters = generate_gabor_filters(size=5, num_filters=16)
model = GaborNetwork(gabor_filters, num_fc_nodes=50).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

experiment_results = []
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    start_time = time()
    train_loss = train(model, device, train_loader, optimizer, epoch)
    accuracy = evaluate_accuracy(model, device, test_loader)
    training_time = time() - start_time
    experiment_results.append({
        'epoch': epoch,
        'accuracy': accuracy,
        'training_time': training_time,
        'loss': train_loss
    })

# Saving the model
torch.save(model.state_dict(), 'gabor_network_model.pth')

# Visualization of results
acc_values = [r['accuracy'] for r in experiment_results]
loss_values = [r['loss'] for r in experiment_results]
time_values = [r['training_time'] for r in experiment_results]
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs, acc_values, '-o')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 3, 2)
plt.plot(epochs, loss_values, '-o')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 3)
plt.plot(epochs, time_values, '-o')
plt.title('Training Time per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training Time (s)')

plt.tight_layout()
plt.show()
