from functools import cache
import numpy as np
import AStar


class Board:

    def __init__(self, board):
        self.board = board
        # self.goal_state_front_blanks, self.goal_state_back_blanks = goal



    def neighbors(self):
        return self
