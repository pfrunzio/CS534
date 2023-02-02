from Algorithm import Algorithm


class AStar(Algorithm):

    def __init__(self, board, heuristic, weighted):
        super().__init__(board, heuristic, weighted)

    def start(self):
        print(self.weighted)
        print(False)
        print(f'Performing A* search with {self.heuristic_type} heuristic {"with" if self.weighted else "without"} weight')
        print("Initial Board:")
        print(self.board)

