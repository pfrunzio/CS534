from Algorithm import Algorithm
import numpy as np

HEURISTIC_TELEPORT = "teleport"
HEURISTIC_SLIDE = "sliding"


class AStar(Algorithm):

    def __init__(self, board, heuristic, weight):
        super().__init__(board)
        self.heuristic = heuristic
        self.weight = weight
        self.goal_state_back_blank, self.goal_state_back_blank = self.goalState(board)

    def start(self):
        print(self.weight)
        print(False)
        print(f'Performing A* search with {self.heuristic} heuristic {"with" if self.weight else "without"} weight')
        print("Initial Board:")
        print(self.board)

    def goalState(self, board):
        arr = []
        num_of_0 = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                if board[x][y] == 0:
                    num_of_0 += 1
                else:
                    arr.append(board[x][y])
        arr.sort()
        goal_state_front_blank = np.insert(arr, 0, [0] * num_of_0).reshape(len(board), len(board[0]))
        goal_state_back_blank = np.append(arr, [0] * num_of_0).reshape(len(board), len(board[0]))
        return goal_state_front_blank, goal_state_back_blank
