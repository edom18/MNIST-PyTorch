import torch
import torch.nn as nn
import torch.nn.functional as F

image_size = 28 * 28
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

checkpoint = torch.load('./model_weights.pth')
model.load_state_dict(checkpoint)

torch.onnx.export(
    model=model,
    args=torch.randn(1, 784),
    f='model.onnx',
    export_params=True,
    input_names=['input'],
    output_names=['output'],
)