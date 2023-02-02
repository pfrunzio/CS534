from abc import abstractmethod, ABC
from functools import cache
from copy import copy, deepcopy

HEURISTIC_TELEPORT = "teleport"
HEURISTIC_SLIDE = "sliding"


class Algorithm(ABC):

    def __init__(self, board):
        self.board = board
        self.goal_state_front_blanks, self.goal_state_back_blanks = self.goal_state(self.board)

    # Driver method for algorithms
    @abstractmethod
    def start(self):
        pass


    # to be run once one init, and never again
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

        back_dict = {}
        front_dict = {}

        for x in range(len(arr)):
            back_dict[arr[x]] = (x // len(board), x % len(board))
            front_dict[arr[x]] = ((x + num_of_0) // len(board), (x + num_of_0) % len(board))

        return front_dict, back_dict

    # Heuristic functions:
    def calculate_heuristic(self, board, heuristic, weighted):
        if heuristic == HEURISTIC_TELEPORT:
            return self._calculate_teleport_heuristic(board, weighted)
        elif heuristic == HEURISTIC_SLIDE:
            return self._calculate_slide_heuristic(board, weighted)
        else:
            return self._calculate_slide_heuristic(board, weighted)

    def _calculate_teleport_heuristic(self, board, weighted):
        front_heuristic = 0
        back_heuristic = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                current = board[x][y]
                if current == 0:
                    continue
                front_blank_coordinates = self.goal_state_front_blanks[current]
                back_blank_coordinates = self.goal_state_back_blanks[current]
                if (x, y) != front_blank_coordinates:
                    front_heuristic += 1 * (current if weighted else 1)
                if (x, y) != back_blank_coordinates:
                    back_heuristic += 1 * (current if weighted else 1)
        return front_heuristic, back_heuristic

    def _calculate_slide_heuristic(self, board, weighted):
        front_heuristic = 0
        back_heuristic = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                current = board[x][y]
                if current == 0:
                    continue
                front_heuristic += self._manhattan_distance_to_goal((x, y), current, True) * (current if weighted else 1)
                back_heuristic += self._manhattan_distance_to_goal((x, y), current, False) * (current if weighted else 1)
        return front_heuristic, back_heuristic

    @cache
    def _manhattan_distance_to_goal(self, location, value, front):
        location_2 = self.goal_state_front_blanks[value] if front else self.goal_state_back_blanks[value]
        return abs(location[0]-location_2[0]) + abs(location[1]-location_2[1])

    # Neighbor functions:
    def neighbors(self, board):
        neighbors = []
        for row in range(len(board)):
            for col in range(len(board)):
                if board[row][col] == 0:
                    neighbors = neighbors + self._neighbor(board, row, col)
        return neighbors

    def _neighbor(self, board, row, col):
        neighbors = []
        map = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for r, c in map:
            if self._is_legal(row + r, col + c) and board[row + r][col + c] != 0:
                new_board = deepcopy(board)
                neighbors.append(self._swap(new_board, row + r, col + c, row, col))
        return neighbors

    def _swap(self, board, row1, col1, row2, col2):
        store = board[row1][col1]
        board[row1][col1] = board[row2][col2]
        board[row2][col2] = store
        return board

    def _is_legal(self, row, col):
        return row >= 0 and col >= 0 and row < len(self.board) and col < len(self.board)

