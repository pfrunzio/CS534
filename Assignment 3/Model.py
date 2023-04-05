from BoardGenerator import extract_board_from_file
import torch
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import r2_score
import time
from AStar import AStar

from Net import Net, PATH, num_epochs, loss_fn, learning_rate, r2_accuracy

TRAIN = False


def train(model, optimizer, loss_fn, x_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


boards3_40 = extract_board_from_file("./Data/ListOfBoards3x3_40_Max_Moves.csv")
boards4_20 = extract_board_from_file("./Data/ListOfBoards4x4_20_Max_Moves.csv")
boards4_60 = extract_board_from_file("./Data/ListOfBoards4x4_60_Max_Moves.csv")
boards5_20 = extract_board_from_file("./Data/ListOfBoards5x5_20_Max_Moves.csv")
boards5_80 = extract_board_from_file("./Data/ListOfBoards5x5_80_Max_Moves.csv")

all_boards = boards3_40 + boards4_20 + boards4_60 + boards5_20 + boards5_80

# fix random seed so the model is consistent between runs
random.seed(99)
random.shuffle(all_boards)

train_percent = 0.8
cut = int(len(all_boards) * train_percent)
    
npuzzle_features = []
npuzzle_cost = []


def generate_features(board):

    manhattan = board.heuristic()

    linear_conflict = board.linear_conflict()

    misplaced_tiles = board.misplaced_tiles()
    
    permutation_inversion = board.permutation_inversion()

    size = len(board)

    features = np.concatenate([np.array([manhattan, linear_conflict, misplaced_tiles, 
                                         permutation_inversion, size])])

    return features, manhattan, misplaced_tiles


npuzzle_features = []
npuzzle_cost = []
manhattan_feature = []
misplaced_tiles_feature = []

for board in all_boards:
    npuzzle_cost.append([board.cost.astype(np.float32)])
    features, manhattan, misplaced_tiles = generate_features(board)
    npuzzle_features.append(features)
    manhattan_feature.append(manhattan)
    misplaced_tiles_feature.append(misplaced_tiles)
    

x_train = torch.tensor(np.array(npuzzle_features[:cut - 1]), dtype=torch.float32)
y_train = torch.tensor(np.array(npuzzle_cost[:cut - 1]), dtype=torch.float32)

x_test = torch.tensor(np.array(npuzzle_features[cut:]), dtype=torch.float32)
y_test = torch.tensor(np.array(npuzzle_cost[cut:]), dtype=torch.float32)

model = Net()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if TRAIN:
    train(model, optimizer, loss_fn, x_train, y_train, num_epochs)
else:
    model.load_state_dict(torch.load(PATH))

torch.save(model.state_dict(), PATH)


# Calculate heuristic speeds

# total_ml = 0
# for board in all_boards:
#     a = AStar(board, "ml", "true")
#     start = time.time()
#     a._calculate_heuristic(board)
#     total_ml += time.time() - start
#
# print(f'ML Speed:\t{total_ml / len(all_boards)}')
#
# total_sliding = 0
# for board in all_boards:
#     a = AStar(board, "sliding", "true")
#     start = time.time()
#     a._calculate_heuristic(board)
#     total_sliding += time.time() - start
#
# print(f'Sliding Speed:\t{total_sliding / len(all_boards)}')


# generate average cost difference data

# cut = 500
# diff = 0
# for board in all_boards[:cut]:
#     a = AStar(board, "sliding", "true")
#     b = AStar(board, "ml", "true")
#     diff += b.start() - a.start()
#
# print(f'Average Diff: {diff / cut}')


# generate model

print("Training: " + str(r2_accuracy(model, x_train, y_train)))
print("Testing: " + str(r2_accuracy(model, x_test, y_test)))
print("Manhattan: " + str(r2_score(np.array(manhattan_feature), np.array(npuzzle_cost))))
print("Misplaced Tiles: " + str(r2_score(np.array(misplaced_tiles_feature), np.array(npuzzle_cost))))
