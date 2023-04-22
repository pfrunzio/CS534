import copy
import random
from copy import deepcopy
from enum import Enum
from Gridworld import Gridworld, Action


def print_geneome(genome):
    string = ""
    for t in range(len(genome)):
        for r in range(len(genome[t])):
            print(genome[t][r])
        print("\n")

class GeneticSequenceTable:
    def __init__(self, gridworld):
        self.gridworld = gridworld

        self.actions = list(Action)
        self.population_size = 1000
        self.num_generations = 100
        self.num_move_per_square = 5

        self.mutation_rate = 0.5
        self.num_parents = 20
        self.num_of_tables =2

    def copy_genome(self, genome):
        if genome is None:
            return None
        new_genome =[]
        table_one = copy.deepcopy(genome[1])
        copy_table_one = copy.deepcopy(table_one)
        new_genome.append(table_one)
        new_genome.append(copy_table_one)
        return new_genome

    def evaluate_genome(self, genome):

        new_gridworld = Gridworld(deepcopy(self.gridworld), self.gridworld.pos, self.gridworld.level)
        if genome is None:
            return 0
        while not new_gridworld.is_terminal:
            action = genome[0][new_gridworld.pos[0]][new_gridworld.pos[1]].pop(0)
            new_gridworld = new_gridworld.take_action(action)
            if not genome[0][new_gridworld.pos[0]][new_gridworld.pos[1]]:
                genome[0][new_gridworld.pos[0]][new_gridworld.pos[1]] = copy.deepcopy(genome[1][new_gridworld.pos[0]][new_gridworld.pos[1]])
        return new_gridworld.turn

    def run(self):
        population = []
        overall_best_genome = None

        for i in range (self.population_size):
            list_of_table = []
            table =[]
            for r in range(len(self.gridworld.gridworld)):
                row = []
                for c in range(len(self.gridworld.gridworld[r])):
                    col = random.choices(self.actions, k=self.num_move_per_square)
                    row.append(col)
                table.append(row)
            list_of_table.append(table)
            copy_table = copy.deepcopy(table)
            list_of_table.append(copy_table)
            population.append(list_of_table)
        print(population[0][0][0][0][0])
        # Iterate over the specified number of generations
        for generation in range(self.num_generations):
            # Evaluate the fitness of each genome
            fitness_scores = [self.evaluate_genome(self.copy_genome(genome)) for genome in population]

            # Print some information about the current generation
            print(f"Generation {generation}: max fitness = {max(fitness_scores)}, avg fitness = {sum(fitness_scores) / len(fitness_scores)}")

            # Output the best genome found by the genetic algorithm
            best_genome = self.copy_genome(population[fitness_scores.index(max(fitness_scores))])

            if self.evaluate_genome(self.copy_genome(best_genome)) >= self.evaluate_genome(self.copy_genome(overall_best_genome)):
                overall_best_genome = self.copy_genome(best_genome)

            # Select the top N genomes as parents for the next generation
            parent_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[
                             :self.num_parents]
            parents = [population[i] for i in parent_indices]

            # Create a new population of genomes through crossover and mutation
            new_population = []

            for i in range(len(parents)):
                new_population.append(self.copy_genome(parents[i]))

            while len(new_population) < self.population_size-self.num_parents:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = []
                table = []
                for r in range(len(self.gridworld.gridworld)):
                    row = []
                    for c in range(len(self.gridworld.gridworld[0])):
                        col = []
                        for _ in range(self.num_move_per_square):
                            col.append(copy.deepcopy(parent1[1][r][c][_]) if random.random() < 0.5 else copy.deepcopy(parent2[1][r][c][_]))
                        row.append(col)
                    table.append(row)
                if len(new_population) < self.population_size/3:
                    for r in range(len(self.gridworld.gridworld)):
                        for c in range(len(self.gridworld.gridworld[r])):
                            for _ in range(self.num_move_per_square):
                                if random.random() < self.mutation_rate:
                                    table[r][c][_] = random.choice(self.actions)
                child.append(table)
                table_copy = copy.deepcopy(table)
                child.append(table_copy)
                new_population.append(child)

        # Replace the old population with the new one
        population = new_population

        # Print some information about the current generation
        print(
           f"Generation {generation}: max fitness = {max(fitness_scores)}, avg fitness = {sum(fitness_scores) / len(fitness_scores)}")
        # Output the best genome found by the genetic algorithm
        best_genome = max(population, key=self.evaluate_genome)

        if self.evaluate_genome(self.copy_genome(best_genome)) >= self.evaluate_genome(
                self.copy_genome(overall_best_genome)):
            overall_best_genome = self.copy_genome(best_genome)

        best_genome_fitness_scores = ([self.evaluate_genome(self.copy_genome(overall_best_genome)) for _ in
                                        range(len(self.population_size))])

        print(
            f"Best overall genome: {self.copy_genome(overall_best_genome)}, average fitness = {sum(best_genome_fitness_scores) / len(best_genome_fitness_scores)}")
        print(
            f"Best genome of Final Generation: {self.copy_genome(best_genome)}, fitness = {self.evaluate_genome(self.copy_genome(best_genome))}")

    def print_geneome(self, genome):
        string = ""
        for t in range(len(genome)):
            for r in range(len(genome[t])):
                print(genome[t][r])
            print("\n")
