from Algorithm import Neighbor, Direction
from copy import copy, deepcopy

class Board:
    def __init__(self, board):
        self.board = board

    def neighbors(self):
        neighbors = []
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                if self.board[row][col] == 0:
                    neighbors = neighbors + self._neighbor(row, col)
        return neighbors

    def _neighbor(self, row, col):
        neighbors = []
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for r, c in directions:
            if self._is_legal(row + r, col + c) and self.board[row + r][col + c] != 0:
                new_board = self._swap(deepcopy(self.board), row + r, col + c, row, col)
                neighbors.append(Neighbor(Board(new_board), new_board[row][col], Direction((-r, -c))))
        return neighbors

    def _swap(self, board, row1, col1, row2, col2):
        store = board[row1][col1]
        board[row1][col1] = board[row2][col2]
        board[row2][col2] = store
        return board

    def _is_legal(self, row, col):
        return row >= 0 and col >= 0 and row < len(self.board) and col < len(self.board)

    def __hash__(self):
        return hash(map(tuple, self.board))

    def __init__(self, board):
        self.board = board

    # allows Board to be treated as a normal 2d array
    def __getitem__(self, item):
        return self.board[item]

    def __iter__(self):
        return self.board.__iter__()

    def __len__(self):
        return len(self.board)

    def __repr__(self):
        return self.board
    def __str__(self):
        return str(self.board)
