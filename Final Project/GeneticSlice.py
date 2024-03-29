import random
from copy import deepcopy
import time

from Gridworld import Gridworld, Action


class GeneticSlice:
    def __init__(self, gridworld, runtime, max_moves):
        self.gridworld = gridworld

        self.actions = list(Action)

        if gridworld.hasInventory == False:
            self.actions.pop()
            self.actions.pop()

        self.population_size = 1000
        self.num_generations = 100

        self.max_moves = max_moves
        self.move_increase_frequency = 5
        self.move_increase = 5

        self.mutation_rate = 0.05

        self.num_parents = round(self.population_size / 10)
        self.num_keep_parents = round(self.num_parents / 10)

        self.runtime = runtime

        self.graph_data = [[], [], [], []]

    def evaluate_genome(self, genome):

        new_gridworld = Gridworld(deepcopy(self.gridworld), self.gridworld.pos, self.gridworld.level)

        for action in genome:

            if new_gridworld.is_terminal:
                return new_gridworld.turn

            new_gridworld = new_gridworld.take_action(action)

        return round(new_gridworld.health / new_gridworld.hunger_lost_per_turn) + new_gridworld.turn

    def run(self):

        print(
            f'Performing Genetic in {self.runtime} seconds\n')

        start_time = time.time()
        end_time = start_time + self.runtime

        # Create a random population of genomes
        num_moves = 0
        population = []

        overall_best_genome = []
        # Iterate over the specified number of generations
        generation = 0

        while time.time() < end_time:

            if generation % self.move_increase_frequency == 0:

                num_moves = min(num_moves + self.move_increase, self.max_moves)

                if generation == 0:
                    population = ([random.choices(self.actions, k=self.move_increase) for _ in
                                   range(self.population_size)])
                elif not (len(population[0]) + self.move_increase) > self.max_moves:

                    for i in range(len(population)):
                        population[i] = population[i] + random.choices(self.actions, k=self.move_increase)

            # Evaluate the fitness of each genome
            fitness_scores = [self.evaluate_genome(genome) for genome in population]

            # Select the top N genomes as parents for the next generation
            parent_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[
                             :self.num_parents]
            parents = [population[i] for i in parent_indices]

            # Create a new population of genomes through crossover and mutation
            new_population = []

            for i in range(self.num_keep_parents):
                new_population.append(parents[i])

            print(len(new_population))

            while len(new_population) < self.population_size:

                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(num_moves)]
                for i in range(num_moves):
                    if random.random() < self.mutation_rate:
                        child[i] = random.choice(self.actions)
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

            generation_data = [generation, max(fitness_scores), sum(fitness_scores) / len(fitness_scores),
                               time.time() - start_time]
            self.graph_data[0].append(generation_data[0])
            self.graph_data[1].append(generation_data[1])
            self.graph_data[2].append(generation_data[2])
            self.graph_data[3].append(generation_data[3])

            # Print some information about the current generation
            print(
                f"Generation {generation}: max fitness = {max(fitness_scores)}, avg fitness = {sum(fitness_scores) / len(fitness_scores)}, time = {time.time() - start_time}")
            generation += 1

        # Output the best genome found by the genetic algorithm
        best_genome = max(population, key=self.evaluate_genome)

        if self.evaluate_genome(best_genome) >= self.evaluate_genome(overall_best_genome):
            overall_best_genome = best_genome

        print(f"Best overall genome: {overall_best_genome}, fitness = {self.evaluate_genome(overall_best_genome)}")
        print(f"Best genome of Final Generation: {best_genome}, fitness = {self.evaluate_genome(best_genome)}")
        return self.graph_data
