from functools import cache
import numpy as np
from copy import copy, deepcopy


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
        return 0

    def neighbors(self):
        neighbors = []
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                if self.board[row][col] == 0:
                    neighbors = neighbors + self._neighbor(row, col)
        return neighbors

    def _neighbor(self, row, col):
        neighbors = []
        map = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for r, c in map:
            if self._isLegal(row + r, col + c) and self.board[row + r][col + c] != 0:
                new_board = deepcopy(self.board)
                neighbors.append(self._swap(new_board, row + r, col + c, row, col))
        return neighbors

    def _swap(self, board, row1, col1, row2, col2):
        store = board[row1][col1]
        board[row1][col1] = board[row2][col2]
        board[row2][col2] = store
        return board

    def _isLegal(self, row, col):
        return row >= 0 and col >= 0 and row < len(self.board) and col < len(self.board)



