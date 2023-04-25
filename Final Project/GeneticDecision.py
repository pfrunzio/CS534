import copy
import random
from copy import deepcopy
from enum import Enum

from DecisionGenome import DecisionGenome
from Gridworld import Gridworld, Action




class GeneticDecision:
    def __init__(self, gridworld):
        self.gridworld = gridworld

        self.actions = list(Action)
        self.movement_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.population_size = 1000
        self.num_generations = 25
        self.child_mutation_rate = 0.2
        self.gene_mutation_rate = 0.2
        self.num_parents = round(self.population_size / 10)
        self.num_keep_parents = round(self.num_parents / 10)
        self.number_of_conditions = 6
        self.number_of_actions = 5
        self.length = self.number_of_conditions + 8

    def evaluate_genome(self, genome):
        list_of_action = []
        new_gridworld = Gridworld(deepcopy(self.gridworld), self.gridworld.pos, self.gridworld.level)
        if genome is None:
            return 0
        while not new_gridworld.is_terminal:
            if not list_of_action or list_of_action is None:
                list_of_action = genome.generate_next_moves(new_gridworld)
            if (list_of_action is not None):
                action = list_of_action.pop(0)
                new_gridworld = new_gridworld.take_action(action)
        return new_gridworld.turn

    def copy_genome(self, genome):
        if genome is None:
            return None
        list_of_condition_number = copy.deepcopy(genome.list_of_conditions)
        list_of_values = copy.deepcopy(genome.list_of_values)
        list_of_actions = copy.deepcopy(genome.list_of_actions)
        default_action = copy.deepcopy(genome.default_action)
        eat_threshold = copy.deepcopy(genome.eat_threshold)
        new_genome = DecisionGenome(list_of_condition_number, list_of_values, list_of_actions, default_action,
                                    eat_threshold)
        return new_genome

    def run(self):

        population = []
        overall_best_genome = None

        for i in range(self.population_size):
            list_of_conditions = random.sample(range(0, self.number_of_conditions), self.number_of_conditions)
            for _ in range(self.length-self.number_of_conditions):
                list_of_conditions.append(random.choice(range(0,self.number_of_conditions)))
            list_of_values = random.choices(range(0, 100), k=self.length)
            list_of_actions = random.choices(range(0, self.number_of_actions), k=self.length)
            default_action = random.choice(self.actions)
            eat_threshold = random.randint(0, 100)
            genome = DecisionGenome(list_of_conditions, list_of_values, list_of_actions, default_action,
                                    eat_threshold)
            population.append(genome)

        # Iterate over the specified number of generations
        for generation in range(self.num_generations):
            # Evaluate the fitness of each genome
            fitness_scores = [self.evaluate_genome(self.copy_genome(genome)) for genome in population]

            # Print some information about the current generation
            print(
                f"Generation {generation}: max fitness = {max(fitness_scores)}, avg fitness = {sum(fitness_scores) / len(fitness_scores)}")
            # Output the best genome found by the genetic algorithm
            best_genome = population[fitness_scores.index(max(fitness_scores))]

            if self.evaluate_genome(self.copy_genome(best_genome)) >= self.evaluate_genome(
                    self.copy_genome(overall_best_genome)):
                overall_best_genome = self.copy_genome(best_genome)

            # Select the top N genomes as parents for the next generation
            parent_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[
                             :self.num_parents]
            parents = [population[i] for i in parent_indices]

            # Create a new population of genomes through crossover and mutation
            new_population = []

            for i in range(len(parents)):
                new_population.append(parents[i])

            while len(new_population) < self.population_size - self.num_parents:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = self.generate_child(parent1, parent2)
                if len(new_population) < self.population_size / 3:
                    if random.random() < self.child_mutation_rate:
                        child = self.mutate(child)
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

        best_genome_fitness_scores = ([self.evaluate_genome(self.copy_genome(overall_best_genome)) for _ in
                                       range(self.population_size)])

        print(
            f"Best overall genome: {self.copy_genome(overall_best_genome)}, average fitness = {sum(best_genome_fitness_scores) / len(best_genome_fitness_scores)}")
        print(
            f"Best genome of Final Generation: {self.copy_genome(best_genome)}, fitness = {self.evaluate_genome(self.copy_genome(best_genome))}")

    def generate_child(self, parent1, parent2):
        list_of_conditions = []
        list_of_values = []
        list_of_actions = []
        default_action = parent1.default_action if random.random() < 0.5 else parent2.default_action
        eat_threshold = random.randint(0, 100)
        for i in range(self.length):
            list_of_conditions.append(self.inherit_number(parent1.list_of_conditions[i], parent2.list_of_conditions[i], True))
            list_of_values.append(self.inherit_number(parent1.list_of_values[i], parent2.list_of_values[i], False))
            list_of_actions.append(self.inherit_number(parent1.list_of_actions[i], parent2.list_of_actions[i], True))
        genome = DecisionGenome(list_of_conditions, list_of_values, list_of_actions, default_action,
                                eat_threshold)
        return genome

    def inherit_number(self, parent1, parent2, in_between):
        inheritance = random.random()
        chance1 = 0.35
        chance2 = 0.35
        if(in_between):
            chance1 = 0.5
            chance2= 0.5
        upper = 0
        lower = 0
        if parent1 > parent2:
            upper = parent1
            lower = parent2
        else:
            upper = parent2
            lower = parent1

        if inheritance < chance1:
            return parent1
        elif inheritance < (chance1+chance2):
            return parent2
        else:
            return random.randint(lower, upper)

    def mutate(self, genome):
        for i in range(self.length):
            if random.random() < self.gene_mutation_rate:
                genome.list_of_actions[i] = random.randint(0, self.number_of_conditions)
            if random.random() < self.gene_mutation_rate:
                genome.list_of_values[i] = random.randint(0, 100)
            if random.random() < self.gene_mutation_rate:
                genome.list_of_actions[i] = random.randint(0, self.number_of_actions)
        if random.random() < self.gene_mutation_rate:
            genome.default_action = random.choice(self.actions)
        if random.random() < self.gene_mutation_rate:
            genome.eat_threshold = random.randint(0, 100)
        return genome
