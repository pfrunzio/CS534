import random
from enum import IntEnum
from enum import Enum
from copy import deepcopy
from queue import Queue

import numpy as np


# Set the tile weights to some arbitrary default values
class TileValue(IntEnum):
    EMPTY = 0
    WATER = 5
    MOUNTAIN = -5
    FOOD = 10
    BOAR = 3
    KILLED_BOAR = 15


class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    # These values are arbitrary
    USE_TILE = (0, 0)
    PICK_UP_ITEM = (-1, -1)
    USE_INVENTORY = (1, 1)


class Inventory(Enum):
    EMPTY = "EMPTY",
    WATER = "WATER",
    FOOD = "FOOD"
    BOAR_MEAT = "BOAR MEAT"


def _tile_char(val):
    if val == TileValue.EMPTY:
        return "-"
    if val == TileValue.WATER:
        return "W"
    if val == TileValue.MOUNTAIN:
        return "M"
    if val == TileValue.FOOD:
        return "F"
    if val == TileValue.BOAR:
        return "B"
    if val == TileValue.KILLED_BOAR:
        return "K"


class Gridworld:
    def __init__(self, gridworld, pos):
        self.gridworld = gridworld
        self.pos = pos
        self.is_terminal = False

        # Feel free to change these values
        self.food_reward = 8
        self.hydration_reward = 7
        self.hunger_lost_per_turn = 5
        self.hydration_lost_per_turn = 4

        self.health = 100
        self.hydration = 100
        self.inventory = Inventory.EMPTY
        self.turn = 1

        self.changes = []

    def pick_ml_action(self, output):
        actions_map = {0: Action.UP, 1: Action.DOWN, 2: Action.LEFT, 3: Action.RIGHT, 4: Action.USE_TILE,
                       5: Action.PICK_UP_ITEM, 6: Action.USE_INVENTORY}

        while True:
            index = output.argmax().item()
            action = actions_map[index]
            if self._is_legal_action(action):
                return action
            else:
                output[index] = -9999999999999

    def _is_legal_action(self, action):
        # TODO: check if an action is legal first
        return True

    def features(self):
        # Features:
        #  - health
        #  - hydration
        #  - has water
        #  - has food
        #  - has boar
        #  - turn number
        #  - position
        #  - past 15 changes

        array = [self.health, self.hydration, self.inventory == Inventory.WATER,
                 self.inventory == Inventory.FOOD, self.inventory == Inventory.BOAR_MEAT,
                 self.turn, self.pos[0], self.pos[1]]

        # append list of changes, only most recent 15
        for i in range(15):
            if i < len(self.changes):
                array.append(self.changes[len(self.changes) - i][0])
                array.append(self.changes[len(self.changes) - i][1])
            else:
                array.append(-1)
                array.append(-1)

        return array

    def get_state(self):
        return self.pos[0], self.pos[1], self.health, self.hydration, self.inventory, self.turn, \
               tuple(sorted(self.changes))

    def _is_legal_pos(self, pos):
        row = pos[0]
        col = pos[1]
        if 0 <= row < len(self.gridworld) and 0 <= col < len(self.gridworld[0]):
            val = self.gridworld[row][col]
            return val != TileValue.MOUNTAIN

        return False

    def take_action(self, action):

        if action == Action.USE_INVENTORY or action == Action.USE_TILE or action == Action.PICK_UP_ITEM:
            return self._non_movement_action(action)

        # Movement actions
        else:

            action = action.value
            rand = random.random()

            new_positions = Queue()

            # Move only if the dehydration debuff is avoided
            if (self.hydration < 50 and rand <= 0.5) or self.hydration >= 50:
                new_positions.put((self.pos[0] + action[0], self.pos[1] + action[1]))
            else:
                new_positions.put(self.pos)

            new_pos = new_positions.get()
            # If the inputted action is not valid, just stay in the same position
            if not self._is_legal_pos(new_pos):
                new_positions.put(self.pos)
                new_pos = new_positions.get()

            return self._move_to(new_pos, self.pos)

    def _non_movement_action(self, action):

        row = self.pos[0]
        col = self.pos[1]
        tile = self.gridworld[row][col]

        new_gridworld = Gridworld(deepcopy(self.gridworld), self.pos)
        new_gridworld.health = deepcopy(self.health)
        new_gridworld.hydration = deepcopy(self.hydration)
        new_gridworld.inventory = deepcopy(self.inventory)
        new_gridworld.turn = deepcopy(self.turn)
        new_gridworld.changes = deepcopy(self.changes)

        # Action for using tile
        if action == Action.USE_TILE:

            if tile == TileValue.FOOD:
                new_gridworld[row][col] = TileValue.EMPTY
                new_gridworld.changes.append((row, col))
                new_gridworld.health = min(new_gridworld.health + new_gridworld.food_reward, 100)

            elif tile == TileValue.KILLED_BOAR:
                new_gridworld[row][col] = TileValue.EMPTY
                new_gridworld.health = min(new_gridworld.health + new_gridworld.food_reward * 2, 100)

            # Water is remains after the agent uses it
            elif tile == TileValue.WATER:
                new_gridworld.hydration = min(new_gridworld.hydration + new_gridworld.hydration_reward, 100)

        # Action for picking up item
        elif action == Action.PICK_UP_ITEM:
            if new_gridworld.gridworld[row][col] == TileValue.KILLED_BOAR:
                new_gridworld.inventory = Inventory.BOAR_MEAT
                new_gridworld.gridworld[row][col] = TileValue.EMPTY
            elif new_gridworld.gridworld[row][col] == TileValue.FOOD:
                new_gridworld.inventory = Inventory.FOOD
                new_gridworld.gridworld[row][col] = TileValue.EMPTY
            elif new_gridworld.gridworld[row][col] == TileValue.WATER:
                new_gridworld.inventory = Inventory.WATER

        # Action for using inventory
        elif action == Action.USE_INVENTORY:
            if new_gridworld.inventory == Inventory.BOAR_MEAT:
                new_gridworld.health = min(self.health + self.food_reward * 2, 100)
            if new_gridworld.inventory == Inventory.FOOD:
                new_gridworld.health = min(new_gridworld.health + new_gridworld.food_reward, 100)
            if new_gridworld.inventory == Inventory.WATER:
                new_gridworld.hydration = min(new_gridworld.hydration + new_gridworld.hydration_reward, 100)
            new_gridworld.inventory = Inventory.EMPTY

        new_gridworld.health = max(new_gridworld.health - new_gridworld.hunger_lost_per_turn, 0)
        new_gridworld.hydration = max(new_gridworld.hydration - new_gridworld.hydration_lost_per_turn, 0)

        if new_gridworld.health <= 0:
            new_gridworld.is_terminal = True

        new_gridworld.turn += 1

        return new_gridworld

    def _move_to(self, new_pos, prev_pos):

        row = new_pos[0]
        col = new_pos[1]
        tile = self.gridworld[row][col]

        new_gridworld = Gridworld(deepcopy(self.gridworld), new_pos)
        new_gridworld.health = deepcopy(self.health)
        new_gridworld.hydration = deepcopy(self.hydration)
        new_gridworld.inventory = deepcopy(self.inventory)
        new_gridworld.turn = deepcopy(self.turn)
        new_gridworld.changes = deepcopy(self.changes)

        # If an agent tries to go on a boar tile, keep it on its current tile if it does not kill the boar
        if tile == TileValue.BOAR:
            # If the tile is in the changes array, we know the agent has already moved to the boar once
            if new_gridworld.changes.__contains__((row, col)):
                new_gridworld[row][col] = TileValue.KILLED_BOAR
            else:
                new_gridworld.changes.append((row, col))
                new_gridworld.health = max(self.health - self.food_reward, 0)
                # Move back to previous tile
                new_gridworld.pos = (prev_pos[0], prev_pos[1])

        new_gridworld.health = max(new_gridworld.health - self.hunger_lost_per_turn, 0)
        new_gridworld.hydration = max(new_gridworld.hydration - self.hydration_lost_per_turn, 0)

        if new_gridworld.health <= 0:
            new_gridworld.is_terminal = True

        new_gridworld.turn += 1

        return new_gridworld

    def __str__(self):
        string = ""
        max_len = len(str(np.amax(self.gridworld)))
        for row, line in enumerate(self.gridworld):
            for col, val in enumerate(line):
                if (row, col) == self.pos and val == TileValue.EMPTY:
                    string += "S".ljust(max_len + 2, " ")
                    continue
                string += _tile_char(val).ljust(max_len + 2, " ")
            string = string.rstrip()
            string += "\n"
        string.rstrip()
        string += f'Agent is on tile ({self.pos[0]},  {self.pos[1]})\n'
        string += "Health: " + self.health.__str__() + "\n"
        string += "Hydration: " + self.hydration.__str__() + "\n"
        string += "Inventory: " + self.inventory.name.__str__() + "\n"
        string += "Turn: " + self.turn.__str__() + "\n"
        return string.rstrip()

    def __getitem__(self, item):
        return self.gridworld[item]

    def __len__(self):
        return len(self.gridworld)

    def replace(self, val, new_val):
        for row in range(len(self.gridworld)):
            for col in range(len(self.gridworld[0])):
                if self.gridworld[row][col] == val:
                    self.gridworld[row][col] = new_val
