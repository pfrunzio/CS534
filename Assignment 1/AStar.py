from Algorithm import Algorithm
from queue import PriorityQueue 


class AStar(Algorithm):

    def __init__(self, board, heuristic, weighted):
        super().__init__(board, heuristic, weighted)

    def start(self):
        print(self.weighted)
        print(False)
        print(f'Performing A* search with {self.heuristic_type} heuristic {"with" if self.weighted else "without"} weight')
        print("Initial Board:")
        print(self.board)

    def search(self):
        fringe = PriorityQueue()
        fringe.put(self.board, 0)
        came_from = dict()
        cost_so_far = dict()
        came_from[self.board] = None
        cost_so_far[self.board] = 0

        while not fringe.empty():
            current = fringe.get()

            if self.calculate_heuristic(self, current) == 0:
                break
            
            for next in self.neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + Algorithm.calculate_heuristic(self, next)
                    fringe.put(next, priority)
                    came_from[next] = current
