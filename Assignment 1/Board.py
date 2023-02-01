from functools import cache
import numpy as np
import AStar
from copy import copy, deepcopy


class Board:

    def __init__(self, board):
        self.board = board

    # allows Board to be treated as a normal 2d array
    def __getitem__(self, item):
        return self.board[item]

    def __iter__(self):
        return self.board.__iter__()

    def __len__(self):
        return len(self.board)

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

