import copy
import random
from copy import deepcopy
from enum import Enum
from Gridworld import Gridworld, Action
import time


def print_geneome(genome):
    string = ""
    for t in range(len(genome)):
        for r in range(len(genome[t])):
            print(genome[t][r])
        print("\n")


class GeneticTable:
    def __init__(self, gridworld, runtime):
        self.gridworld = gridworld

        self.actions = list(Action)

        if gridworld.hasInventory == False:
            self.actions.pop()
            self.actions.pop()

        self.movement_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.population_size = 1000
        self.num_generations = 100

        self.mutation_rate = 0.05

        self.num_parents = round(self.population_size / 10)
        self.num_keep_parents = round(self.num_parents / 10)

        self.num_of_tables = 2
        
        self.runtime = runtime
        self.graph_data = [[], [], [], []]
    def evaluate_genome(self, genome):

        new_gridworld = Gridworld(deepcopy(self.gridworld), self.gridworld.pos, self.gridworld.level)
        if genome is None:
            return 0
        while not new_gridworld.is_terminal:
            action = genome[0][new_gridworld.pos[0]][new_gridworld.pos[1]]
            if action == Action.USE_INVENTORY or action == Action.USE_TILE or action == Action.PICK_UP_ITEM:
                genome[0][new_gridworld.pos[0]][new_gridworld.pos[1]] = genome[2][new_gridworld.pos[0]][new_gridworld.pos[1]]
            new_gridworld = new_gridworld.take_action(action)
        return round(new_gridworld.health / new_gridworld.hunger_lost_per_turn) + new_gridworld.turn

    def copy_genome(self, genome):
        if genome is None:
            return None
        new_genome =[]
        table_one = copy.deepcopy(genome[1])
        copy_table_one = copy.deepcopy(table_one)
        table_two = copy.deepcopy(genome[2])
        new_genome.append(table_one)
        new_genome.append(copy_table_one)
        new_genome.append(table_two)
        return new_genome

    def run(self):

        population = []
        overall_best_genome = None

        for i in range (self.population_size):
            list_of_table = []
            table_one = ([random.choices(self.actions, k=len(self.gridworld.gridworld[0])) for _ in
                       range(len(self.gridworld.gridworld))])
            table_two = ([random.choices(self.movement_actions, k=len(self.gridworld.gridworld[0])) for _ in
                       range(len(self.gridworld.gridworld))])
            list_of_table.append(table_one)
            copy_table_one = copy.deepcopy(table_one)
            list_of_table.append(copy_table_one)
            list_of_table.append(table_two)
            population.append(list_of_table)

        start_time = time.time()
        end_time = start_time + self.runtime
        
        generation = 1

        while generation < self.num_generations and time.time() < end_time:
            # Evaluate the fitness of each genome
            fitness_scores = [self.evaluate_genome(self.copy_genome(genome)) for genome in population]

            # Print some information about the current generation
            print(
                f"Generation {generation}: max fitness = {max(fitness_scores)}, avg fitness = {sum(fitness_scores) / len(fitness_scores)}, , time = {time.time()-start_time}")
            # Output the best genome found by the genetic algorithm
            generation_data = [generation, max(fitness_scores), sum(fitness_scores) / len(fitness_scores),
                               time.time() - start_time]
            self.graph_data[0].append(generation_data[0])
            self.graph_data[1].append(generation_data[1])
            self.graph_data[2].append(generation_data[2])
            self.graph_data[3].append(generation_data[3])

            best_genome = self.copy_genome(population[fitness_scores.index(max(fitness_scores))])

            if self.evaluate_genome(self.copy_genome(best_genome)) >= self.evaluate_genome(
                    self.copy_genome(overall_best_genome)):
                overall_best_genome = self.copy_genome(best_genome)

            # Select the top N genomes as parents for the next generation
            parent_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[
                             :self.num_parents]
            parents = [population[i] for i in parent_indices]

            # Create a new population of genomes through crossover and mutation
            new_population = []

            for i in range(self.num_keep_parents):
                new_population.append(self.copy_genome(parents[i]))

            while len(new_population) < self.population_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = []
                table_one = []
                table_two = []
                for r in range(len(self.gridworld.gridworld)):
                    row_one = []
                    row_two = []
                    for c in range(len(self.gridworld.gridworld[r])):
                        row_one.append(copy.deepcopy(parent1[1][r][c]) if random.random() < 0.5 else copy.deepcopy(
                            parent2[1][r][c]))
                        row_two.append(copy.deepcopy(parent1[2][r][c]) if random.random() < 0.5 else copy.deepcopy(
                            parent2[2][r][c]))
                    table_one.append(row_one)
                    table_two.append(row_two)
                if len(new_population) < self.population_size/3:
                    for r in range(len(self.gridworld.gridworld)):
                        for c in range(len(self.gridworld.gridworld[r])):
                            if random.random() < self.mutation_rate:
                                table_one[r][c] = random.choice(self.actions)
                                table_two[r][c] = random.choice(self.movement_actions)
                child.append(copy.deepcopy(table_one))
                copy_table_one = deepcopy(table_one)
                child.append(copy_table_one)
                child.append(table_two)
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population
            generation += 1

        # Print some information about the current generation
        print( f"Generation {generation}: max fitness = {max(fitness_scores)}, avg fitness = {sum(fitness_scores) / len(fitness_scores)}")
        # Output the best genome found by the genetic algorithm
        generation_data = [generation, max(fitness_scores), sum(fitness_scores) / len(fitness_scores),
                           time.time() - start_time]
        self.graph_data[0].append(generation_data[0])
        self.graph_data[1].append(generation_data[1])
        self.graph_data[2].append(generation_data[2])
        self.graph_data[3].append(generation_data[3])

        best_genome = max(population, key=self.evaluate_genome)

        if self.evaluate_genome(self.copy_genome(best_genome)) >= self.evaluate_genome(self.copy_genome(overall_best_genome)):
            overall_best_genome = self.copy_genome(best_genome)

        best_genome_fitness_scores = ([self.evaluate_genome(self.copy_genome(overall_best_genome)) for _ in
                       range(self.population_size)])

        print(f"Best overall genome: {self.copy_genome(overall_best_genome)}, average fitness = {sum(best_genome_fitness_scores)/len(best_genome_fitness_scores)}")
        print(f"Best genome of Final Generation: {self.copy_genome(best_genome)}, fitness = {self.evaluate_genome(self.copy_genome(best_genome))}")
        return self.graph_data

