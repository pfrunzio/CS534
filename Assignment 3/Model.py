from BoardGenerator import extract_board_from_file 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.dropout = nn.Dropout(.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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

boards = extract_board_from_file("./Data/CorrectListOfBoards3x3.csv")
    
npuzzle_features = []
npuzzle_cost = []
    
def generate_features(board):

    manhattan = board.heuristic()

    linear_conflict = board.linear_conflict()

    misplaced_tiles = board.misplaced_tiles()
    
    permutation_inversion = board.permutation_inversion()

    features = np.concatenate([np.array([manhattan, linear_conflict, misplaced_tiles, permutation_inversion]), 
                               np.array([])])

    return features

for board in boards:
    npuzzle_cost.append([board.cost.astype(np.float32)])
    features = generate_features(board)
    npuzzle_features.append(features)

x_train = torch.tensor(np.array(npuzzle_features[:10000]), dtype=torch.float32)
y_train = torch.tensor(np.array(npuzzle_cost[:10000]), dtype=torch.float32)

x_test = torch.tensor(np.array(npuzzle_features[10000:]), dtype=torch.float32)
y_test = torch.tensor(np.array(npuzzle_cost[10000:]), dtype=torch.float32)

# Define the hyperparameters
input_size = 4 #num of features
hidden_size1 = 10
hidden_size2 = 20
hidden_size3 = 40
output_size = 1 #path cost
learning_rate = .1
num_epochs = 200

model = Net()

# define the loss function
loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, optimizer, loss_fn, x_train, y_train, num_epochs)

PATH = "./Data/net.pth"
torch.save(model.state_dict(), PATH)

# model = Net()
# model.load_state_dict(torch.load(PATH))

print(r2_accuracy(model, x_test, y_test))
