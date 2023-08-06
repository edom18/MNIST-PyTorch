import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Hyper Parameters
num_epochs = 10
num_batch = 100
learning_rate = 0.001
image_size = 28 * 28

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor()
])

# Preparing training and test data

# Training data
train_data = MNIST(
    './datasets/mnist',
    train=True,
    download=True,
    transform=transform,
)
train_loader = DataLoader(
    train_data,
    batch_size=num_batch,
    shuffle=True,
)

# Test data
test_data = MNIST(
    './datasets/mnist',
    train=False,
    download=False,
    transform=transform,
)
test_loader = DataLoader(
    test_data,
    batch_size=num_batch,
    shuffle=True,
)

class Net(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()

    self.l1 = nn.Linear(input_size, 100) # From a input layer to a hidden layer.
    self.l2 = nn.Linear(100, output_size) # From a hidden layer to a output layer.

  def forward(self, x):
    x = self.l1(x)
    x = torch.sigmoid(x)
    x = self.l2(x)
    return F.log_softmax(x, dim=1)

# Create a neural network.
model = Net(image_size, 10).to(device)

# Loss function
loss_func = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Traning
model.train() # Change mode to training

for epoch in range(num_epochs):
  loss_sum = 0

  for inputs, labels in train_loader:
    # Send data to GPU if it can do
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Initialize optimizer
    optimizer.zero_grad()

    # Perform the neural network.
    inputs = inputs.view(-1, image_size)
    outputs = model(inputs)

    # Calculate loss
    loss = loss_func(outputs, labels)
    loss_sum += loss

    # Calculate gradiation
    loss.backward()

    # Update its weights
    optimizer.step()

  # Show the progress
  print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_loader)}')

  # Save its weights
  torch.save(model.state_dict(), 'model_weights.pth')

# Evaluation
model.eval() # Change mode to eval

loss_sum = 0
correct = 0

with torch.no_grad():
  for inputs, labels in test_loader:

    # Send data to GPU if it can do
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Perform the neural network.
    inputs = inputs.view(-1, image_size)
    outputs = model(inputs)

    loss_sum += loss_func(outputs, labels)

    pred = outputs.argmax(1)
    correct += pred.eq(labels.view_as(pred)).sum().item()

  print(f'Loss: {loss_sum.item() / len(test_loader)}, Accuracy: {100 * correct / len(test_data)}% ({correct}/{len(test_data)})')