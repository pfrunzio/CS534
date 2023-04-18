from Gridworld import Gridworld, Action
from copy import deepcopy
from Net import Net, PATH, num_epochs, learning_rate


def train(model, optimizer, loss_fn, x_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = evaluate(model)
        loss = loss_fn(y_pred, y_train) # change this to something
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


def evaluate(self, model):

    new_gridworld = Gridworld(deepcopy(self.gridworld), self.gridworld.pos)

    while not new_gridworld.is_terminal:
        action = 0  # TODO
        new_gridworld = new_gridworld.take_action(action)

    return round(new_gridworld.health / new_gridworld.hunger_lost_per_turn) + new_gridworld.turn











