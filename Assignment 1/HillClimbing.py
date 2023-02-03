import random

from Algorithm import Algorithm
from Algorithm import HEURISTIC_SLIDE
import time

MAX_SIDEWAYS_MOVES = 7


class HillClimbing(Algorithm):

    def __init__(self, board, seconds):
        super().__init__(board, HEURISTIC_SLIDE, False)
        self.seconds = seconds

    def start(self):
        print(f'Performing greedy (hill climbing) search for {self.seconds} seconds')
        print("Initial Board:")
        print(self.board)

    def random_restart(self, enable_sideways):

        end_time = time.time() + self.seconds

        best_board = self.board
        best_cost = min(self.calculate_heuristic(self.board))
        count = 0
        while time.time() < end_time:
            current_board, current_cost = self.greedy_hill_climbing(enable_sideways)
            print(current_board)
            print(current_cost)
            if current_cost < best_cost:
                best_board = current_board
                best_cost = current_cost
            count += 1
        print(best_board)
        print(best_cost)
        print("Iterations: ", count)
        return best_board, best_cost

    def greedy_hill_climbing(self, enable_sideways):

        local_min = False

        count = 0
        sideways_move_count = 0
        new_board = self.board
        current_cost = min(self.calculate_heuristic(self.board))

        while not local_min:
            current_cost = min(self.calculate_heuristic(new_board))
            best_neighbor, best_neighbor_score = self._get_best_neighbor(enable_sideways, new_board)

            if best_neighbor_score < current_cost:
                new_board = best_neighbor
            elif best_neighbor_score == current_cost and sideways_move_count > MAX_SIDEWAYS_MOVES:
                new_board = best_neighbor
                sideways_move_count += 1
            else:
                local_min = True

            count += 1

        return new_board, current_cost

    # TODO: if heuristic = 0, return immediately?
    def _get_best_neighbor(self, enable_sideways, current_board):

        current_board_cost = min(self.calculate_heuristic(current_board))

        neighbors = self.neighbors(current_board)

        sideways_neighbors = []
        tied_neighbors = []

        best_neighbor_score = min(self.calculate_heuristic(neighbors[0].board))

        for neighbor in neighbors:
            neighbor_score = min(self.calculate_heuristic(neighbor.board))

            if neighbor_score < best_neighbor_score:
                tied_neighbors = [(neighbor.board, neighbor_score)]

                best_neighbor_score = neighbor_score
            elif neighbor_score == best_neighbor_score:
                tied_neighbors.append((neighbor.board, neighbor_score))

            if neighbor_score == current_board_cost:
                sideways_neighbors.append((neighbor.board, neighbor_score))

        if enable_sideways and best_neighbor_score > current_board_cost and len(sideways_neighbors) > 0:
            return random.choice(sideways_neighbors)

        return random.choice(tied_neighbors)
