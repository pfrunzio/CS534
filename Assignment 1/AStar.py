from Algorithm import Algorithm
from queue import PriorityQueue 


class AStar(Algorithm):

    def __init__(self, board, heuristic, weighted):
        super().__init__(board, heuristic, weighted)
        self.nodes_expanded = 0

    def start(self):
        print(self.weighted)
        print(False)
        print(f'Performing A* search with {self.heuristic_type} heuristic {"with" if self.weighted else "without"} weight')
        print("Initial Board:")
        end = self.search()

        print("\nPath:")
        path = self._create_path_string(end)
        print("\n".join(path))

        print("\nCost: {}".format(end.cost))
        print("Length: {}".format(len(path)))

        print("Nodes Expanded: {}".format(self.nodes_expanded))
        print("Estimated Branching Factor: {}".format(pow(self.nodes_expanded, 1 / len(path))))

        print()

    def search(self):
        print(self.board)

        fringe = PriorityQueue()
        fringe.put((self.calculate_heuristic(self.board), self.board))
        visited = set()

        while not fringe.empty():
            current = fringe.get()[1]
            visited.add(current)
            self.nodes_expanded += 1

            if self.calculate_heuristic(current) == 0:
                return current

            for board, value, direction in current.neighbors():
                if board not in visited:
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
