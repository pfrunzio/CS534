import numpy as np
from Algorithm import Direction
from copy import deepcopy
from AStar import AStar


class Board:
    def __init__(self, board, previous=None, cost=0, direction=None, value=None):
        self.board = board
        self.previous = previous
        self.cost = cost
        self.direction = direction
        self.value = value

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
                board = Board(new_board, self, self.cost + new_board[row][col], Direction((-r, -c)), new_board[row][col])
                neighbors.append(board)
        return neighbors

    def _swap(self, board, row1, col1, row2, col2):
        board[row1][col1], board[row2][col2] = board[row2][col2], board[row1][col1]
        return board

    def _is_legal(self, row, col):
        return 0 <= row < len(self.board) and 0 <= col < len(self.board)

    def __hash__(self):
        string = ""
        for row in self.board:
            for num in row:
                string += str(num)
        return hash(string)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return (self.board == other.board).all()

    def __lt__(self, other):
        np.less(self.board, other.board)

    # allows Board to be treated as a normal 2d array
    def __getitem__(self, item):
        return self.board[item]

    def __iter__(self):
        return self.board.__iter__()

    def __len__(self):
        return len(self.board)

    def __str__(self):
        return '\n'.join(map(str, self.board))
    
    def heuristic(board):
        b = AStar(board, "sliding", "true")
        return b._calculate_heuristic(board)

    def linear_conflict(board):
        n = len(board)
        conflict_count = 0
        for i in range(n):
            for j in range(n):
                if board[i][j] != 0 and (i * n + j + 1) % n != 0:
                    for k in range(j + 1, n):
                        if board[i][j] > board[i][k] and (i * n + k + 1) % n != 0:
                            conflict_count += 1
                    for k in range(i + 1, n):
                        if board[i][j] > board[k][j] and (k * n + j + 1) % n != 0:
                            conflict_count += 1
        return 2 * conflict_count
    
    def misplaced_tiles(board):
        n = len(board)
        count = 0
        for i in range(n):
            for j in range(n):
                if board[i][j] != 0 and board[i][j] != i * n + j + 1:
                    count += 1
        return count
    
    def permutation_inversion(board):
        # Flatten the board into a 1D array
        flattened_board = [num for row in board for num in row]
        
        # Calculate the permutation inversion count
        inversions = 0
        for i in range(len(flattened_board)):
            if flattened_board[i] == 0: # Ignore blank tile
                continue
            for j in range(i+1, len(flattened_board)):
                if flattened_board[j] == 0: # Ignore blank tile
                    continue
                if flattened_board[i] > flattened_board[j]:
                    inversions += 1
                    
        return inversions