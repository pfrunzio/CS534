import torch
import torch.nn as nn
import torch.nn.functional as F
from Net import Net, PATH, num_epochs, learning_rate, loss_fn
import torch
import torch.optim as optim
import numpy as np
from RL import RL
from Main import get_gridworld
from Gridworld import Gridworld, Action, Inventory
import math


# Define the hyperparameters
input_size = 36  # 9 + 7 + 10 * 2
hidden_size = 20
output_size = 1
learning_rate = .1
num_epochs = 50

PATH = "./Data/net.pth"


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# use linear loss
loss_fn = nn.MSELoss()


def state_to_input(q_entry):
    state = q_entry[0]
    action = q_entry[1]

    input = [state[0], state[1], state[2], state[3], state[5]]

    for i in list(Inventory):
        input.append(i == state[4])

    for a in list(Action):
        input.append(action == a)

    count = 0
    for row, col in state[6]:
        if count < 10:
            input.append(row)
            input.append(col)
        count += 1

    for i in range(count, 10):
        input.append(-1)
        input.append(-1)

    return input


class Agent:
    def __init__(self, gridworld):
        self.gridworld = gridworld

        self.gamma = .99
        self.epsilon = 0.5
        self.policy_net = Net()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.rl = RL(gridworld, 15)
        self.rl.start()

    def train(self):
        states = list(self.rl.q_table.keys())

        q_values = []
        for state in states:
            q_values.append(self.rl.q_table[state])

        inputs = [state_to_input(state) for state in states]

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            y_values = torch.tensor(np.array(q_values), dtype=torch.float32).unsqueeze(1)
            y_pred = self.policy_net(torch.tensor(inputs, dtype=torch.float32))

            loss = loss_fn(y_pred, y_values)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1} Loss: {loss.item():.4f}')


gridworld = get_gridworld("Level2.txt", 2)

agent = Agent(gridworld)

agent.train()


class DQN:
    def __init__(self, gridworld, net):
        self.gridworld = gridworld

        self.actions = list(Action)

        if gridworld.hasInventory == False:
            self.actions.pop()
            self.actions.pop()

        self.net = net

    def start(self):
        print(
            f'Performing RL in {self.runtime} seconds\n')
        print("Initial World:")
        print(self.gridworld, '\n')

        # start learning
        return self._rl()

    def _get_best_action(self, state):

        best_utility = -math.inf
        best_action = Action.UP

        for action in self.actions:

            current_utility = self.net(torch.tensor(state_to_input((state, action)), dtype=torch.float32))

            if current_utility > best_utility:
                best_utility = current_utility
                best_action = action

        return best_action

    def calc_mean_reward(self):
        trial_count = 0
        total_reward = 0

        # average over 100 runs
        while trial_count < 100:

            terminal = False
            current_gridworld = self.gridworld
            current_state = current_gridworld.get_q_table_state()

            trial_reward = 0

            while not terminal:
                action = self._get_best_action(current_state)

                new_board = current_gridworld.take_action(action)
                reward = 1
                terminal = new_board.is_terminal

                trial_reward += reward
                current_gridworld = new_board
                current_state = current_gridworld.get_q_table_state()

            total_reward += trial_reward
            trial_count += 1

        return total_reward / trial_count


dqn = DQN(gridworld, agent.policy_net)

print(dqn.calc_mean_reward())

