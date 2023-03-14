from enum import IntEnum
from enum import Enum
from queue import Queue
from copy import deepcopy
import random
import numpy as np


class Value(IntEnum):
    EMPTY = 0
    COOKIE = -10
    GLASS = -11
    BARRIER = -12
    UP_ARROW = -13
    DOWN_ARROW = -14
    LEFT_ARROW = -15
    RIGHT_ARROW = -16


class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Gridworld:
    def __init__(self, gridworld, position):
        self.gridworld = gridworld
        self.position = position

        # list of (row, col) pairs of cookies/switches the agent has landed on
        self.changes = []

    def get_state(self):
        return self.position[0], self.position[1], tuple(sorted(self.changes))

    def take_action(self, action, move_reward, transition_model):
        action = action.value
        rand = random.random()

        # list of where it's trying to move and backup options if it can't in order of priority
        new_positions = Queue()

        if rand < transition_model:
            # tries to move forward one square (intended result)
            new_positions.put((self.position[0] + action[0], self.position[1] + action[1]))
        elif rand < 1 - ((1 - transition_model) / 2):
            # tries to move forward two squares
            new_positions.put((self.position[0] + 2 * action[0], self.position[1] + 2 * action[1]))
            new_positions.put((self.position[0] + action[0], self.position[1] + action[1]))
        else:
            # tries to move back one square
            new_positions.put((self.position[0] + -1 * action[0], self.position[1] + -1 * action[1]))

        # final backup: if it can't move at all, it stays in current position
        new_positions.put(self.position)

        # figure out position to move to
        new_position = new_positions.get()
        while not self._is_legal_position(new_position):
            new_position = new_positions.get()

        return self._move_to(new_position, move_reward)

    def _move_to(self, new_position, move_reward):

        row = new_position[0]
        col = new_position[1]

        # calculate reward and new board based on action
        new_board = Gridworld(deepcopy(self.gridworld), new_position)
        new_board.changes = deepcopy(self.changes)
        reward = move_reward
        terminal = False

        landed_on = self.gridworld[row][col]

        # eat cookie or glass
        if landed_on == Value.COOKIE:
            new_board[row][col] = Value.EMPTY
            new_board.changes.append((row, col))
            reward += 2
        elif landed_on == Value.GLASS:
            new_board[row][col] = Value.EMPTY
            new_board.changes.append((row, col))
            reward -= 2
        # land on switch
        elif ord('a') <= -landed_on <= ord('z'):
            new_board[row][col] = Value.EMPTY
            char = chr(-landed_on).upper()
            new_board.replace(-ord(char), Value.EMPTY)
            new_board.changes.append((row, col))
        # reached terminal state
        elif landed_on != 0:
            terminal = True
            reward += landed_on

        return new_board, reward, terminal

    def _is_legal_position(self, position):
        row = position[0]
        col = position[1]
        if 0 <= row < len(self.gridworld) and 0 <= col < len(self.gridworld[0]):
            val = self.gridworld[row][col]
            if ord('A') <= -val <= ord('Z'):
                return False
            return val != Value.BARRIER

        return False

    def __str__(self):
        string = ""
        max_len = len(str(np.amax(self.gridworld)))
        for row, line in enumerate(self.gridworld):
            for col, val in enumerate(line):
                if (row, col) == self.position and val == Value.EMPTY:
                    string += "S".ljust(max_len + 2, " ")
                    continue
                # make value 1 longer than max characters long
                string += self._value_str(val).ljust(max_len + 2, " ")
            string = string.rstrip()
            string += "\n"
        return string.rstrip()

    def _value_str(self, value):
        if value == Value.EMPTY:
            return "0"
        if value == Value.COOKIE:
            return "+"
        if value == Value.GLASS:
            return "-"
        if value == Value.BARRIER:
            return "X"
        if value == Value.UP_ARROW:
            return "^"
        if value == Value.DOWN_ARROW:
            return "v"
        if value == Value.LEFT_ARROW:
            return "<"
        if value == Value.RIGHT_ARROW:
            return ">"
        if ord("A") <= -value <= ord("z"):
            return chr(-value)
        
        return str(value)

    def __getitem__(self, item):
        return self.gridworld[item]

    def __len__(self):
        return len(self.gridworld)

    def replace(self, val, new_val):
        for row in range(len(self.gridworld)):
            for col in range(len(self.gridworld[0])):
                if self.gridworld[row][col] == val:
                    self.gridworld[row][col] = new_val
