from Gridworld import Gridworld, Action
from copy import deepcopy
from Net import Net, PATH, num_epochs, learning_rate, loss_fn
import torch
import torch.optim as optim
import random
from Main import get_gridworld
import numpy as np


class Agent:
    def __init__(self, gridworld):
        self.gridworld = gridworld

        self.gamma = .99
        self.epsilon = 0.5
        self.policy_net = Net()
        self.target_net = Net()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
    def update(self, batch_size):
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            score = torch.tensor(np.array([self.evaluate(self.policy_net)]), dtype=torch.float32, requires_grad=True)
            best_score = torch.tensor(np.array([100000]), dtype=torch.float32)

            loss = loss_fn(score, best_score)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1} Loss: {loss.item():.4f}')

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
            input = torch.tensor(np.array(features), dtype=torch.float32)
            action = new_gridworld.pick_ml_action(model(input))

            new_gridworld = new_gridworld.take_action(action)

        return new_gridworld.turn


gridworld = get_gridworld("mountain_pass.txt")

agent = Agent(gridworld)

agent.update(100)

