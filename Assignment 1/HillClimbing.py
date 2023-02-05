import math
import random

import matplotlib.pyplot as plt

from Algorithm import Algorithm
from Algorithm import HEURISTIC_SLIDE
import time

MIN_BOARD_SIZE = 3
MAX_SIDEWAYS_MOVES = 7
MAX_GREEDY_ITERATIONS = 0
INITIAL_TEMP_BASE = 4
TEMP_MODIFIER_BASE = 5


class HillClimbing(Algorithm):

    def __init__(self, board, seconds):
        super().__init__(board, HEURISTIC_SLIDE, False)
        self.seconds = seconds

    def start(self):
        print(f'Performing greedy (hill climbing) search for {self.seconds} seconds')
        print("Initial Board:")
        print(self.board)
        print(self.random_restart(True))

    def random_restart(self, enable_sideways):

        end_time = time.time() + self.seconds

        best_board = self.board
        best_cost = min(self.calculate_heuristic(self.board))
        count = 0

        moves = []
        total_neighbor_count = 0
        total_cost = 0
        greedy_counter = 1

        while time.time() < end_time:

            if greedy_counter <= MAX_GREEDY_ITERATIONS:
                current_board, current_cost, move_list, neighbor_count = self.greedy_hill_climbing(enable_sideways)

                # greedy_counter += 1
            else:
                time_diff = end_time - time.time()
                current_board, current_cost, move_list, neighbor_count = self.hill_climbing_annealing(best_board, time_diff, count)

            # print(current_board)
            # print(current_cost)
            if current_cost < best_cost:
                best_board = current_board
                best_cost = current_cost
                total_cost += best_cost
                moves.extend(move_list)

            count += 1
            total_neighbor_count += neighbor_count

            if best_cost == 0:
                break
        # print(best_board)
        # print(best_cost)
        # print("Iterations: ", count)
        return best_board, best_cost, total_neighbor_count, total_cost, moves

    def greedy_hill_climbing(self, enable_sideways):

        local_min = False

        count = 0
        sideways_move_count = 0
        new_board = self.board
        current_cost = min(self.calculate_heuristic(self.board))

        move_list = []

        total_neighbor_count = 0

        while not local_min:
            current_cost = min(self.calculate_heuristic(new_board))
            best_neighbor, best_neighbor_score, neighbor_count = self._get_best_neighbor(enable_sideways, new_board)
            best_neighbor_board = best_neighbor.board
            total_neighbor_count += neighbor_count

            if best_neighbor_score < current_cost:
                new_board = best_neighbor_board
                move_list.append((best_neighbor.value, best_neighbor.direction))
            elif best_neighbor_score == current_cost and sideways_move_count > MAX_SIDEWAYS_MOVES:
                new_board = best_neighbor_board
                move_list.append((best_neighbor.value, best_neighbor.direction))
                sideways_move_count += 1
            else:
                local_min = True

            count += 1

        return new_board, current_cost, move_list, total_neighbor_count

    # TODO: if heuristic = 0, return immediately?
    def _get_best_neighbor(self, enable_sideways, current_board):

        neighbor_count = 0

        current_board_cost = min(self.calculate_heuristic(current_board))

        neighbors = self.neighbors(current_board)
        neighbor_count += len(neighbors)

        sideways_neighbors = []
        tied_neighbors = []

        best_neighbor_score = min(self.calculate_heuristic(neighbors[0].board))

        for neighbor in neighbors:
            neighbor_score = min(self.calculate_heuristic(neighbor.board))

            if neighbor_score < best_neighbor_score:
                tied_neighbors = [(neighbor, neighbor_score)]

                best_neighbor_score = neighbor_score
            elif neighbor_score == best_neighbor_score:
                tied_neighbors.append((neighbor, neighbor_score))

            if neighbor_score == current_board_cost:
                sideways_neighbors.append((neighbor, neighbor_score))

        if enable_sideways and best_neighbor_score > current_board_cost and len(sideways_neighbors) > 0:
            return random.choice(sideways_neighbors)

        return random.choice(tied_neighbors), neighbor_count

    def hill_climbing_annealing(self, current_board, time_diff, iteration_num):

        new_board = current_board

        board_size = len(new_board)
        size_ratio = MIN_BOARD_SIZE / board_size

        temp_modifier_constant = size_ratio * min(TEMP_MODIFIER_BASE, 1 + (
                TEMP_MODIFIER_BASE / time_diff))  # decrease in constant == increase in random probability

        initial_temp_constant = max(1, INITIAL_TEMP_BASE - (
                INITIAL_TEMP_BASE / time_diff))  # decrease == decrease in random probability

        # TODO: Change stepcount based on time
        max_time_step = 1000

        current_time_step = 1

        initial_temp = initial_temp_constant

        probabilities = []

        costs = []

        move_list = []
        total_neighbor_count = 0

        while current_time_step <= max_time_step:
            current_cost = min(self.calculate_heuristic(new_board))

            if current_cost == 0:
                break

            neighbors = self.neighbors(new_board)
            total_neighbor_count += len(neighbors)

            chosen_neighbor = random.choice(neighbors)
            chosen_neighbor_board = chosen_neighbor.board

            neighbor_cost = min(self.calculate_heuristic(chosen_neighbor_board))

            costs.append(current_cost)

            if neighbor_cost <= current_cost:
                new_board = chosen_neighbor_board
                move_list.append((chosen_neighbor.value, chosen_neighbor.direction))
                move_probability = 1
            else:
                move_probability = math.exp((current_cost - neighbor_cost) / self._calculate_temperature(
                    current_time_step,
                    initial_temp, temp_modifier_constant))
                if random.random() <= move_probability:
                    new_board = chosen_neighbor_board
                    move_list.append((chosen_neighbor.value, chosen_neighbor.direction))

            print(current_time_step)
            print(move_probability)

            probabilities.append(move_probability)

            current_time_step += 1

        # self._graph_temperature(probabilities, list(range(1, max_time_step + 1)), iteration_num)
        # self._graph_cost_vs_time(costs, list(range(1, max_time_step + 1)), iteration_num)
        return new_board, current_cost, move_list, total_neighbor_count

    def _calculate_temperature(self, time_step, initial_temp, temp_constant):
        return initial_temp / math.log(time_step + temp_constant)

    def _graph_temperature(self, probabilities, time_steps, iteration):
        plt.scatter(time_steps, probabilities)
        plt.ylabel("Move Probability")
        plt.xlabel("Time Step")
        plt.title(f'Probability vs Time (Iteration: {iteration})')
        plt.show()

    def _graph_cost_vs_time(self, costs, time_steps, iteration):
        plt.scatter(time_steps, costs)
        plt.ylabel("Board Cost")
        plt.xlabel("Time Step")
        plt.title(f'Cost vs Time (Iteration: {iteration})')
        plt.show()
