from abc import abstractmethod, ABC
from functools import cache

import AStar


class Algorithm(ABC):

    def __init__(self, board):
        self.board = board
        self.goal_state_front_blanks, self.goal_state_back_blanks = self.goal_state(self.board)

    # Driver method for algorithms
    @abstractmethod
    def start(self):
        pass


    def calculate_heuristic(self, heuristic, weighted):
        if heuristic == AStar.HEURISTIC_TELEPORT:
            return self.calculate_teleport_heuristic(self.board, weighted)
        elif heuristic == AStar.HEURISTIC_SLIDE:
            return self.calculate_slide_heuristic(self.board, weighted)
        else:
            return self.calculate_slide_heuristic(self.board, weighted)

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

    def calculate_teleport_heuristic(self, board, weighted):
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

    def calculate_slide_heuristic(self, board, weighted):
        front_heuristic = 0
        back_heuristic = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                current = board[x][y]
                if current == 0:
                    continue
                front_heuristic += self.manhattan_distance_to_goal((x, y), current, True) * (current if weighted else 1)
                back_heuristic += self.manhattan_distance_to_goal((x, y), current, False) * (current if weighted else 1)
        return front_heuristic, back_heuristic

    @cache
    def manhattan_distance_to_goal(self, location, value, front):
        location_2 = self.goal_state_front_blanks[value] if front else self.goal_state_back_blanks[value]
        return abs(location[0]-location_2[0]) + abs(location[1]-location_2[1])
