from abc import abstractmethod, ABC
from functools import cache

import AStar


class Algorithm(ABC):

    def __init__(self, board):
        self.board = board
        self.goal_state_front_blanks, self.goal_state_back_blanks = self.goal_state(self.board)
        print(self.goal_state_front_blanks)
        print(self.goal_state_back_blanks)

    # Driver method for algorithms
    @abstractmethod
    def start(self):
        pass


    def calculate_heuristic(self, heuristic, weighted):
        if heuristic == AStar.HEURISTIC_TELEPORT:
            return self.calculate_teleport_heuristic(self.board, self.goal_state_front_blanks, self.goal_state_back_blanks, weighted)
        elif heuristic == AStar.HEURISTIC_SLIDE:
            return self.calculate_slide_heuristic(self.board, self.goal_state_front_blanks, self.goal_state_back_blanks, weighted)
        else:
            return self.calculate_slide_heuristic(self.board, self.goal_state_front_blanks, self.goal_state_back_blanks, weighted)


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

    def calculate_teleport_heuristic(self, board, goal_state_front_blank, goal_state_back_blank, weighted):
        front_heuristic = 0
        back_heuristic = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                current = board[x][y]
                if current == 0:
                    continue
                front_blank_coordinates = goal_state_front_blank[current]
                back_blank_coordinates = goal_state_back_blank[current]
                if (x, y) != front_blank_coordinates:
                    front_heuristic += 1 * (current if weighted else 1)
                if (x, y) != back_blank_coordinates:
                    back_heuristic += 1 * (current if weighted else 1)
        return front_heuristic, back_heuristic

    def calculate_slide_heuristic(self, board, goal_state_front_blank, goal_state_back_blank, weighted):
        front_heuristic = 0
        back_heuristic = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                current = board[x][y]
                if current == 0:
                    continue
                front_blank_coordinates = goal_state_front_blank[current]
                back_blank_coordinates = goal_state_back_blank[current]
                print(current)
                front_heuristic += self.manhattan_distance((x, y), front_blank_coordinates) * (current if weighted else 1)
                back_heuristic += self.manhattan_distance((x, y), back_blank_coordinates) * (current if weighted else 1)
        return front_heuristic, back_heuristic

    def manhattan_distance(self, location_1, location_2):
        print(location_1)
        print(location_2)
        print(location_2[0])
        return abs(location_1[0]-location_2[0]) + abs(location_1[1]-location_2[1])