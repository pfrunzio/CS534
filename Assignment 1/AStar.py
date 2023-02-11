from Algorithm import Algorithm
from Algorithm import HEURISTIC_SLIDE
from queue import PriorityQueue
from HillClimbing import HillClimbing
from Board import Board
import time


class AStar(Algorithm):

    def __init__(self, board, heuristic, weighted):
        super().__init__(board, heuristic, weighted)
        self.nodes_expanded = 0
        self.elapsed_time = None
        self.greedy_cache = dict()

    def start(self):
        print(f'Performing A* search with {self.heuristic_type} heuristic {"with" if self.weighted else "without"} weight')
        print("Initial Board:")

        print(self.board)
        end = self.search()

        self._print_path(end)

    def search(self):

        fringe = PriorityQueue()
        fringe.put((self._calculate_heuristic(self.board), self.board))
        visited = dict()

        start_time = time.time()
        while not fringe.empty():
            current = fringe.get()[1]

            self.nodes_expanded += 1

            if self._calculate_heuristic(current) == 0:
                self.elapsed_time = time.time() - start_time
                return current

            for board, value, direction in current.neighbors():
                if board not in visited or board.cost < visited[board]:
                    visited[current] = current.cost
                    priority = board.cost + self._calculate_heuristic(board)
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

    # Part 3 Code:
    def _calculate_greedy_heuristic(self, board):
        copy = Board(board.board)
        # local_min, cost, _, _, _ = HillClimbing(copy, False, 0).greedy_hill_climbing(False, copy)
        local_min = self._hill_climbing(copy)
        if type(local_min) is not Board:
            return local_min

        cost = local_min.cost
        rest_cost = self._greedy_a_star(Board(local_min.board), 10, 15)

        # caching values
        current = local_min
        while current.previous is not None:
            self.greedy_cache[current] = local_min.cost - current.cost + rest_cost
            current = current.previous

        return cost + rest_cost

    def _hill_climbing(self, board):

        if board in self.greedy_cache:
            return self.greedy_cache[board]

        current = board
        while True:
            neighbors = [n[0] for n in current.neighbors()]
            neighbors.sort(key=lambda n: self.calculate_heuristic(n), reverse=False)
            best = neighbors[0]
            if self.calculate_heuristic(best) >= self.calculate_heuristic(current):
                return current
            current = best

    def _greedy_a_star(self, board, weight, node_limit):

        if board in self.greedy_cache:
            return self.greedy_cache[board]

        fringe = PriorityQueue()
        fringe.put((self.calculate_heuristic(board), board))
        visited = dict()

        nodes_explored = 0

        while not fringe.empty():
            current = fringe.get()[1]
            nodes_explored += 1

            if nodes_explored >= node_limit or self.calculate_heuristic(current) == 0:
                final = current
                heuristic = self.calculate_heuristic(final)

                # caching values
                while current.previous is not None:
                    self.greedy_cache[current] = final.cost - current.cost + heuristic
                    current = current.previous

                return final.cost + heuristic

            for board, value, direction in current.neighbors():
                if board not in visited or board.cost < visited[board]:
                    visited[current] = current.cost
                    priority = board.cost + (weight * self.calculate_heuristic(board))
                    fringe.put((priority, board))
