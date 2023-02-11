from Algorithm import Algorithm
from Algorithm import HEURISTIC_SLIDE
from queue import PriorityQueue
from HillClimbing import HillClimbing
from Board import Board
import time


class AStar(Algorithm):

    def __init__(self, board, heuristic, weighted, heuristic_weight=1, time_limit=None):
        super().__init__(board, heuristic, weighted)
        self.nodes_expanded = 0
        self.elapsed_time = None
        self.heuristic_weight = heuristic_weight
        self.time_limit = time_limit

    def start(self):
        print(f'Performing A* search with {self.heuristic_type} heuristic {"with" if self.weighted else "without"} weight')
        print("Initial Board:")

        print(self.board)
        end = self.search()

        self._print_path(end)

    def search(self):

        fringe = PriorityQueue()
        fringe.put((self.calculate_heuristic(self.board), self.board))
        visited = dict()

        start_time = time.time()
        while not fringe.empty():
            current = fringe.get()[1]
            if self.time_limit is not None and time.time() - start_time > self.time_limit:
                return current

            #if self.heuristic_type == "greedy":
            #    print(self.nodes_expanded)
            self.nodes_expanded += 1

            if self._calculate_heuristic(current) == 0:
                self.elapsed_time = time.time() - start_time
                return current

            for board, value, direction in current.neighbors():
                if board not in visited or board.cost < visited[board]:
                    visited[current] = current.cost
                    priority = board.cost + (self.heuristic_weight * self._calculate_heuristic(board))
                    fringe.put((priority, board))

    def _create_path_string(self, goal):
        current = goal
        moves = []

        while current.previous is not None:
            moves.append("{} {}".format(current.value, current.direction.name))
            current = current.previous

        moves.reverse()
        return moves

    def _print_path(self, end):
        path = self._create_path_string(end)

        print("\nPath:")
        print("\n".join(path))

        print("\nFinal Board State:")
        print(end)

        print("\nCost: {}".format(end.cost))
        print("Length: {}".format(len(path)))

        print("Nodes Expanded: {}".format(self.nodes_expanded))
        print("Estimated Branching Factor: {}".format(pow(self.nodes_expanded, 1 / len(path))))
        print("Elapsed Time: {}".format(self.elapsed_time))

        print()

    def _calculate_heuristic(self, board):
        return self.calculate_heuristic(board)\
            if (self.heuristic_type != "greedy") \
            else self._calculate_greedy_heuristic(board)

    def _calculate_greedy_heuristic(self, board):
        copy = Board(board.board)
        local_min, current_cost, _, _, _ = HillClimbing(copy, False, 0).greedy_hill_climbing(False, copy)
        local_min.cost = 0

        a_star = AStar(local_min, HEURISTIC_SLIDE, self.weighted, 10, 0.001)
        new_board = a_star.search()
        rest_cost = new_board.cost + a_star._calculate_heuristic(new_board)
        return current_cost + rest_cost
