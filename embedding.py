from network import MyNetwork
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# (1) generate the MNIST network (you should import your code from task 1),

model = MyNetwork()

# (2) read an existing model from a file and load the pre-trained weights, 

model.load_state_dict(torch.load('./mnist_model.pth'))

# (3) freeze the network weights, and 
for param in model.parameters():
    param.requires_grad = False

# do not freeze the last layer to train the model
model.fc2 = nn.Linear(50, 3)

# (4) replace the last layer with a new Linear layer with three nodes. 
# greek data set transform

class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# load training data set
training_set_path = "./greek_train"
greek_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder( training_set_path,
                                        transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                    GreekTransform(),
                                                                                    torchvision.transforms.Normalize(
                                                                                        (0.1307,), (0.3081,) ) ] ) ),
    batch_size = 5,
    shuffle = True )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Initialize lists to keep track of losses and the number of examples seen
train_losses = []
test_losses = []
examples_count = []

# Training loop for a specified number of epochs
for epoch in range(1, 3 + 1):
    model.train()

    for batch_idx, (data, target) in enumerate(greek_train):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # Use negative log likelihood loss
        loss.backward()
        optimizer.step()

        # Record training loss and number of examples seen
        if batch_idx % 3 == 0:
            train_losses.append(loss.item())
            examples_count.append((epoch - 1) * len(greek_train.dataset) + batch_idx * len(data))

print('train_losses', train_losses)
print('test_losses', test_losses)
print('examples_count', examples_count)