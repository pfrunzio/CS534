import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

# Define the hyperparameters
input_size = 5  # num of features
hidden_size = 10
output_size = 1  # path cost
learning_rate = .1
num_epochs = 200

PATH = "./Data/net.pth"


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# define the loss function
loss_fn = nn.MSELoss()

def r2_accuracy(model, x_test, y_test):
    with torch.no_grad():
        y_pred = model(x_test)
        r2_acc = r2_score(y_test.numpy(), y_pred.numpy())
    return r2_acc