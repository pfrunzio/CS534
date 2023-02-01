from functools import cache
import numpy as np


class Board:

    def __init__(self, board):
        self.board = board

    @cache
    def heuristic(self, sliding, weighted):
        return 0;

    def neighbors(self):
        return self

    @cache
    def goal_state(self, board):
        arr = []
        num_of_0 = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                if board[x][y] == 0:
                    num_of_0 += 1
                else:
                    arr.append(board[x][y])
        arr.sort()

        # Heuristic of Back Blank

        back_dict = {}
        front_dict = {}

        for x in range(len(arr)):
            back_dict[arr[x]] = (x // len(board), x % len(board))
            front_dict[arr[x]] = ((x + num_of_0) // len(board), (x + num_of_0) % len(board))

    def calculate_teleport_heuristic(self, board):
        return 0;
