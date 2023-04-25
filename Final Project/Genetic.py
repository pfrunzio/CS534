import random
from copy import deepcopy
import time

from Gridworld import Gridworld, Action


class Genetic:
    def __init__(self, gridworld, runtime):
        self.gridworld = gridworld
                
        self.actions = list(Action)
        
        if gridworld.hasInventory == False:
            self.actions.pop()
            self.actions.pop()
        
        self.population_size = 1000
        self.num_generations = 100
        self.num_turns = 50

        self.mutation_rate = 0.2
        self.num_parents = 25
        
        self.runtime = runtime

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
        self.end_time = end_time
        
        population = ([random.choices(self.actions, k=self.num_turns) for _ in
                        range(self.population_size)])
        
        generation = 1

        while time.time() < end_time:
            
            # Evaluate the fitness of each genome
            fitness_scores = [self.evaluate_genome(genome) for genome in population]

            # Select the top N genomes as parents for the next generation
            parent_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[
                            :self.num_parents]
            parents = [population[i] for i in parent_indices]

            # Create a new population of genomes through crossover and mutation
            new_population = []
            while len(new_population) < self.population_size:
                
                new_population.append(parents[0])

                parent1 = random.choice(parents[:500])
                parent2 = random.choice(parents[:500])
                child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(self.num_turns)]
                for i in range(self.num_turns):
                    if random.random() < self.mutation_rate:
                        child[i] = random.choice(self.actions)
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population
            # Print some information about the current generation
            print(
                f"Generation {generation}: max fitness = {max(fitness_scores)}, avg fitness = {sum(fitness_scores) / len(fitness_scores)}, time = {time.time()-start_time}")
            generation += 1

        # Output the best genome found by the genetic algorithm
        best_genome = max(population, key=self.evaluate_genome)
        print(f"Best genome: {best_genome}, fitness = {self.evaluate_genome(best_genome)}")

