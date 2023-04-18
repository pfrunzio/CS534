from Gridworld import Gridworld, Action
from copy import deepcopy
from Net import Net, PATH, num_epochs, learning_rate
import torch
import torch.optim as optim
import random


class Agent:
    def __init__(self):
        self.gamma = .99
        self.epsilon = 1
        self.policy_net = Net()
        self.target_net = Net()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
    def update(self, batch_size):
        self.optimizer.zero_grad()
        loss = 0  # change this to evaluate loss based on how long the agent survives
        loss.backward()
        self.optimizer.step()

        print('Loss: {:.4f}'.format(loss.item()))

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # select a random action
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                _, action = q_values.max(1)
                return action.item()


def evaluate(self, model):

    new_gridworld = Gridworld(deepcopy(self.gridworld), self.gridworld.pos)

    while not new_gridworld.is_terminal:
        features = new_gridworld.features()
        action = new_gridworld.pick_ml_action(model(features))
        new_gridworld = new_gridworld.take_action(action)

    return new_gridworld.turn
