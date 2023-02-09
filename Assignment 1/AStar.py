from Algorithm import Algorithm
from queue import PriorityQueue
from HillClimbing import HillClimbing
import time


class AStar(Algorithm):

    def __init__(self, board, heuristic, weighted):
        super().__init__(board, heuristic, weighted)
        self.nodes_expanded = 0
        self.elapsed_time = None

    def start(self):
        print(f'Performing A* search with {self.heuristic_type} heuristic {"with" if self.weighted else "without"} weight')
        print("Initial Board:")

        end = self.search()

        self._print_path(end)

    def search(self):
        print(self.board)

        fringe = PriorityQueue()
        fringe.put((self.calculate_heuristic(self.board), self.board))
        visited = dict()

        start_time = time.time()
        while not fringe.empty():
            current = fringe.get()[1]
            self.nodes_expanded += 1

            heuristic = \
                self.calculate_heuristic(current) \
                if (self.heuristic_type != "greedy") \
                else self._calculate_greedy_heuristic(current)

            if heuristic == 0:
                self.elapsed_time = time.time() - start_time
                return current

            for board, value, direction in current.neighbors():
                if board not in visited or board.cost < visited[board]:
                    visited[current] = current.cost
                    priority = board.cost + self.calculate_heuristic(board)
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

    def _calculate_greedy_heuristic(self, board):
        greedy = HillClimbing(board, self.weighted, 0)
        local_min, current_cost, _, _ = greedy.greedy_hill_climbing(False, board)
        return current_cost + self.calculate_heuristic(local_min)
