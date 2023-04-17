from Gridworld import Gridworld, Action
from copy import deepcopy


class RL:

    def __init__(self, gridworld):
        self.gridworld = gridworld

    def evaluate(self, model):

        new_gridworld = Gridworld(deepcopy(self.gridworld), self.gridworld.pos)

        while not new_gridworld.is_terminal:
            action = 0  # TODO
            new_gridworld = new_gridworld.take_action(action)

        return round(new_gridworld.health / new_gridworld.hunger_lost_per_turn) + new_gridworld.turn




