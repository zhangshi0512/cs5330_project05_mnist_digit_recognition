# Shi Zhang
# evaluate_model.py loads the neural network model and evaluates it on a test dataset

# Import statements
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from network import MyNetwork

# Load the test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=10, shuffle=False)

# Initialize the network and load trained model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth')) 
model.eval()  # Set the model to evaluation mode

# Process the first batch of test images
data, target = next(iter(test_loader))
output = model(data)
_, predicted = torch.max(output, 1)

# Print network output, predicted and actual labels
for i in range(len(data)):
    print(f"Image {i + 1}")
    formatted_output = [f"{value:.2f}" for value in output[i].data.numpy()]
    print("Network Output:", formatted_output)
    print("Predicted Label:", predicted[i].item())
    print("Actual Label:", target[i].item())
    print()

# Plot the first 9 images
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axs.flatten()):
    if i < 9:
        ax.imshow(data[i][0], cmap='gray', interpolation='none')
        ax.set_title(f"Predicted: {predicted[i].item()}")
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()
