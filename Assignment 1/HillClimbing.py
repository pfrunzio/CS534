import math
import random

import matplotlib.pyplot as plt

from Algorithm import Algorithm, Move
from Algorithm import HEURISTIC_SLIDE
import time

MIN_BOARD_SIZE = 3
MAX_SIDEWAYS_MOVES = 7
INITIAL_TEMP_BASE = 4
TEMP_MODIFIER_BASE = 5


class HillClimbing(Algorithm):

    def __init__(self, board, weighted, seconds):
        super().__init__(board, HEURISTIC_SLIDE, weighted)
        self.seconds = seconds

    def start(self):
        print(f'Performing greedy (hill climbing) search for {self.seconds} seconds')
        print("Initial Board:")
        print(self.board.board)
        best_board, cost, total_neighbor_count, moves = self.random_restart()
        self.print_solution(best_board, cost, total_neighbor_count, moves)

    def random_restart(self):

        end_time = time.time() + self.seconds

        best_board = self.board
        best_cost = self.calculate_heuristic(self.board.board)
        count = 0

        solution = None

        moves = []
        total_neighbor_count = 0

        solution_cost = math.inf

        while time.time() < end_time:
            time_diff = end_time - time.time()
            current_board, current_cost, move_list, neighbor_count = self.hill_climbing_annealing(best_board,
                                                                                                  time_diff, True,
                                                                                                  INITIAL_TEMP_BASE,
                                                                                                  TEMP_MODIFIER_BASE)
            if current_cost < best_cost:
                best_board = current_board
                best_cost = current_cost
                moves.extend(move_list)

            count += 1
            total_neighbor_count += neighbor_count

            if best_cost == 0:
                if best_board.cost < solution_cost:
                    solution_cost = best_board.cost
                    solution = best_board.board, best_board.cost, total_neighbor_count, moves
                best_board = self.board
                best_cost = self.calculate_heuristic(self.board.board)
                count = 0
                moves = []

        if solution is None:
            return best_board.board, best_board.cost, total_neighbor_count, moves

        return solution

    def greedy_hill_climbing(self, enable_sideways, board):

        local_min = False

        count = 0
        sideways_move_count = 0
        new_board = board
        current_cost = self.calculate_heuristic(new_board.board)

        move_list = []

        total_neighbor_count = 0

        while not local_min:

            current_cost = self.calculate_heuristic(new_board.board)

            best_neighbor, neighbor_count = self._get_best_neighbor(enable_sideways, new_board)
            best_neighbor_board = best_neighbor[0].board
            best_neighbor_score = self.calculate_heuristic(best_neighbor_board)
            total_neighbor_count += 1

            if best_neighbor_score < current_cost:
                new_board = best_neighbor_board
                move_list.append(Move(best_neighbor_board.value, best_neighbor_board.direction))
                if best_neighbor_score == 0:
                    local_min = True
            elif best_neighbor_score == current_cost and sideways_move_count > MAX_SIDEWAYS_MOVES:
                new_board = best_neighbor_board
                move_list.append(Move(best_neighbor.value, best_neighbor.direction))
                sideways_move_count += 1
            else:
                local_min = True

            count += 1

        return new_board, current_cost, move_list, total_neighbor_count

    # TODO: if heuristic = 0, return immediately?
    def _get_best_neighbor(self, enable_sideways, current_board):

        neighbor_count = 0

        current_board_cost = self.calculate_heuristic(current_board)

        neighbors = current_board.neighbors()

        sideways_neighbors = []
        tied_neighbors = []

        best_neighbor_score = self.calculate_heuristic(neighbors[0].board)

        for neighbor in neighbors:
            neighbor_score = self.calculate_heuristic(neighbor.board)
            neighbor_count += 1

            if neighbor_score < best_neighbor_score:
                tied_neighbors = [(neighbor, neighbor_score)]
                best_neighbor_score = neighbor_score
                if neighbor_score == 0:
                    break
            elif neighbor_score == best_neighbor_score:
                tied_neighbors.append((neighbor, neighbor_score))

            if neighbor_score == current_board_cost:
                sideways_neighbors.append((neighbor, neighbor_score))

        if enable_sideways and best_neighbor_score > current_board_cost and len(sideways_neighbors) > 0:
            return random.choice(sideways_neighbors), neighbor_count

        return random.choice(tied_neighbors), neighbor_count

    def hill_climbing_annealing(self, current_board, time_diff, modified_temp_constants, temp_base, temp_modifier):

        new_board = current_board

        board_size = len(new_board.board)
        size_ratio = MIN_BOARD_SIZE / board_size

        if time_diff <= 0:
            time_diff = .1

        if modified_temp_constants:
            temp_modifier_constant = size_ratio * min(temp_modifier, 1 + (
                    temp_modifier / time_diff))  # decrease in constant == increase in random probability

            initial_temp_constant = max(1, temp_base - (
                    temp_base / time_diff))  # decrease == decrease in random probability
        else:
            temp_modifier_constant = temp_modifier
            initial_temp_constant = temp_base

        # TODO: Change stepcount based on time
        max_time_step = 1000

        current_time_step = 1

        initial_temp = initial_temp_constant

        current_cost = self.calculate_heuristic(new_board.board)

        probabilities = []

        move_list = []
        total_neighbor_count = 0

        while current_time_step <= max_time_step:
            current_cost = self.calculate_heuristic(new_board.board)

            if current_cost == 0:
                break

            neighbors = new_board.neighbors()
            total_neighbor_count += 1

            chosen_neighbor = random.choice(neighbors)
            chosen_neighbor_board = chosen_neighbor.board

            neighbor_cost = self.calculate_heuristic(chosen_neighbor_board.board)

            if neighbor_cost <= current_cost:
                new_board = chosen_neighbor_board
                move_list.append(Move(chosen_neighbor.value, chosen_neighbor.direction))
                move_probability = 1
            else:
                move_probability = math.exp((current_cost - neighbor_cost) / self._calculate_temperature(
                    current_time_step,
                    initial_temp, temp_modifier_constant))
                if random.random() <= move_probability:
                    new_board = chosen_neighbor_board
                    move_list.append(Move(chosen_neighbor.value, chosen_neighbor.direction))

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

    def graph_greedy_vs_annealing_vs_time(self, board, title):

        annealing = self._graph_time_helper(False, board, INITIAL_TEMP_BASE, TEMP_MODIFIER_BASE, True)
        greedy = self._graph_time_helper(True, board, INITIAL_TEMP_BASE, TEMP_MODIFIER_BASE, True)

        print(annealing[0])

        plt.scatter(annealing[0], annealing[1], c='b', marker='x', label='Annealing')
        plt.scatter(greedy[0], greedy[1], c='r', marker='o', label='Greedy')
        plt.legend(loc='upper left')
        plt.ylabel("Heuristic Cost")
        plt.xlabel("Time")
        plt.title(
            f'Heuristic Cost vs Time {title} {"(Weighted Heuristic)" if self.weighted else "(Unweighted Heuristic)"}')
        plt.show()

    def graph_annealing_temp_vs_time(self, board, title):

        annealing1 = self._graph_time_helper(False, board, 10, TEMP_MODIFIER_BASE, False)
        annealing2 = self._graph_time_helper(False, board, 5, TEMP_MODIFIER_BASE, False)
        annealing3 = self._graph_time_helper(False, board, 1, TEMP_MODIFIER_BASE, False)

        plt.scatter(annealing1[0], annealing1[1], c='b', marker='x', label='Initial Temp: 10')
        plt.scatter(annealing2[0], annealing2[1], c='r', marker='o', label='Initial Temp: 5')
        plt.scatter(annealing3[0], annealing3[1], c='g', marker='^', label='Initial Temp: 1')

        plt.legend(loc='upper left')
        plt.ylabel("Heuristic Cost")
        plt.xlabel("Time")
        plt.title(
            f'Heuristic Cost vs Time w/ Initial Temp\n{title} {"(Weighted Heuristic)" if self.weighted else "(Unweighted Heuristic)"}')
        plt.show()

    def _graph_time_helper(self, greedy, board, temp_base, temp_mod, modified_temp_constant):

        best_board = board
        best_cost = self.calculate_heuristic(self.board.board)

        solution = None

        moves = []
        total_neighbor_count = 0

        solution_cost = math.inf

        heuristic_cost_time = ([], [])
        end_time = time.time() + self.seconds
        start_time = time.time()
        while time.time() < end_time:
            current_time = time.time()
            time_diff = end_time - time.time()

            if greedy:
                current_board, current_cost, move_list, neighbor_count = self.greedy_hill_climbing(True, board)
            else:
                current_board, current_cost, move_list, neighbor_count = self.hill_climbing_annealing(best_board,
                                                                                                      time_diff, modified_temp_constant,
                                                                                                      temp_base,
                                                                                                      temp_mod)

            heuristic_cost_time[1].append(current_cost)
            heuristic_cost_time[0].append(current_time - start_time)

            if current_cost < best_cost:
                best_board = current_board
                best_cost = current_cost
                moves.extend(move_list)

            total_neighbor_count += neighbor_count

            if best_cost == 0:
                if best_board.cost < solution_cost:
                    solution_cost = best_board.cost
                    solution = best_board.board, best_board.cost, total_neighbor_count, moves
                best_board = self.board
                best_cost = self.calculate_heuristic(self.board.board)
                moves = []
                total_neighbor_count = 0

        if solution is None:
            return heuristic_cost_time

        return heuristic_cost_time
