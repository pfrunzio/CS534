import torch.nn as nn
import torch.nn.functional as F

PATH = "./Data/net.pth"
input_size = 2
hidden_size = 5
output_size = 1
learning_rate = 0.1
num_epochs = 1000


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
