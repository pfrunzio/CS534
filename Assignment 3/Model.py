from BoardGenerator import extract_board_from_file 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import numpy as np
from AStar import AStar

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, optimizer, loss_fn, x_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

def r2_accuracy(model, x_test, y_test):
    with torch.no_grad():
        y_pred = model(x_test)
        r2_acc = r2_score(y_test.numpy(), y_pred.numpy())
    return r2_acc

boards = extract_board_from_file("./Data/ListOfBoards3x3.csv")

def heuristic(board):
    b = AStar(board, "sliding", "true")
    return b._calculate_heuristic(board)
    
    
x_train_array = []
y_train_array = []

for board in boards:
    x_train_array.append([heuristic(board).astype(np.float32)])
    y_train_array.append([board.cost.astype(np.float32)])

x_train = torch.tensor(x_train_array)
y_train = torch.tensor(y_train_array)

x_test = torch.tensor(x_train_array)
y_test = torch.tensor(y_train_array)

# Define the hyperparameters
input_size = x_train_array.__len__()
hidden_size = 1
output_size = y_train_array.__len__()
learning_rate = 0.1
num_epochs = 3

model = Net(input_size, hidden_size, output_size)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train(model, optimizer, loss_fn, x_train, y_train, num_epochs)

PATH = "./Data/net.pth"
torch.save(model.state_dict(), PATH)

# model = Net(input_size, hidden_size, output_size)
# model.load_state_dict(torch.load(PATH))

print(r2_accuracy(model, x_test, y_test))
