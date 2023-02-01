from Algorithm import Algorithm


HEURISTIC_TELEPORT = "teleport"
HEURISTIC_SLIDE = "sliding"


class AStar(Algorithm):

    def __init__(self, board, heuristic, weight):
        super().__init__(board)
        self.heuristic = heuristic
        self.weight = weight

    def start(self):
        print(self.weight)
        print(False)
        print(f'Performing A* search with {self.heuristic} heuristic {"with" if self.weight else "without"} weight')
        print("Initial Board:")
        print(self.board)

