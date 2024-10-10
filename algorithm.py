# Erin Rainville - 40179308
# Program for algorithm

import random
from deap import base, creator, tools, algorithms
import fitness
from typing import List, Tuple
from multiprocessing import Pool

#Files
INPUT_FILENAME = "Ass1Input.txt"
OUTPUT_FILENAME = "OutputFiles/Ass1Output.txt"

# Input file: read and create a string list of the available pieces
def read_input_file() -> List[str]:
    pieces_list = []
    with open(INPUT_FILENAME) as file:
        for line in file.readlines():
            piece = line.split(' ')
            for column in piece:
                pieces_list.append(column.strip())
    return pieces_list

# Initial puzzle: with defined pieces that are in random positions and orientations
def set_initial_puzzle(pieces_list: List[str]) -> List[str]:
    random.shuffle(pieces_list)
    return [rotate_piece(piece) for piece in pieces_list]

# Rotation: keep order and just change starting point (preserves sides)
def rotate_piece(piece: str) -> str:
    start = random.randint(0, len(piece) - 1)
    rotated = ''.join(piece[(start + i) % len(piece)] for i in range(len(piece)))
    return rotated

# Crossover: selects a random block within puzzle and swaps it between parents
def block_crossover(parent1, parent2, block_size):
    row_size, col_size = 8, 8
    # the start of the block !> than the puzzle size - the block size, else block will be out of bounds
    start_row, start_col = random.randint(0, row_size - block_size), random.randint(0, col_size - block_size)
    # for each piece in defined block, trade piece between parents
    for i in range(block_size):
        for j in range(block_size):
            idx = (start_row + i) * col_size + (start_col + j)
            parent1[idx], parent2[idx] = parent2[idx], parent1[idx]
    return parent1, parent2

# Probability of crossover: depends on the change in the best fitness over generations:
    # probability is defined in algorithms.eaMuCommaLambda
def dynamic_cxpb_adaptive(prev_fitness, current_fitness, cxpb):
    if current_fitness < prev_fitness:
        return max(0.1, cxpb * 0.9)
    else:
        return min(1.0, cxpb * 1.1)

# Mutation: randomly replaces pieces in puzzle with random piece in Ass1Input puzzle, by a given probability MUTPB
def mutate_puzzle(puzzle: List[str], pieces, mutpb) -> tuple[list[str]]:
    for i in range(len(puzzle)):
        if random.random() < mutpb:
            new_piece = random.choice(pieces)
            rotated_piece = rotate_piece(new_piece)
            puzzle[i] = rotated_piece
    return puzzle,

# Probability of mutation: like probability of crossover
def dynamic_mutpb_adaptive(prev_fitness, current_fitness, mutpb):
    if current_fitness < prev_fitness:
        return max(0.01, mutpb * 0.9)
    else:
        return min(1.0, mutpb * 1.1)

# Calculate fitness: for one puzzle,  calls main from fitness.py
def calculate_fitness(puzzle: List[str]) -> Tuple[int]:
    return (fitness.main(puzzle),)

# Prevent stagnation: replace some puzzles with new puzzles initialized with Ass1Input puzzle
def increase_diversity(population, toolbox, diversity_rate):
    num_to_replace = int(len(population) * diversity_rate)
    new_individuals = [toolbox.individual() for _ in range(num_to_replace)]
    population[-num_to_replace:] = new_individuals

# Output: write output in correct format
def write_output_file(puzzle):
    with open(OUTPUT_FILENAME, 'w') as file:
        file.write('Erin Rainville - 40179308')
        for i in range(0, len(puzzle), 8):
            row = ' '.join(puzzle[i:i+8])
            file.write('\n' + row)

# DEAP setup
def set_deap_framework(block_size, tournament_size, mutpb):
    # Get all puzzle pieces available
    pieces = read_input_file()

    # Set up type
    if hasattr(creator, 'FitnessMin'): # for grid search
        del creator.FitnessMin
    if hasattr(creator, 'Individual'): # for grid search
        del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # minimization problem
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialization: assign toolbox attributes to defined functions
    toolbox = base.Toolbox()
    toolbox.register("attr_puzzle", set_initial_puzzle, pieces)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_puzzle)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operators: assign toolbox operators to defined functions
    toolbox.register("evaluate", calculate_fitness)
    toolbox.register("mate", block_crossover, block_size = block_size)
    toolbox.register("mutate", mutate_puzzle, pieces = pieces, mutpb = mutpb)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size) # survivor selection

    return toolbox

# Migration: periodically exchanged puzzles between islands (populations) to maintain diversity
def migrate(populations, migration_rate):
    num_migrants = int(len(populations) * migration_rate)
    for i in range(len(populations)):
        next_island = (i + 1) % len(populations) # last population gives to first
        migrants = random.sample(populations[i], num_migrants)
        populations[next_island].extend(migrants)
        populations[i] = [ind for ind in populations[i] if ind not in migrants] # remove migrants from current population

# Islands: main function that controls the algorithm for an island
def run_island(toolbox, population_size, ngen, LAMBDA, cxpb, mutpb, island_id, num_islands, stagnation_threshold, diversity_rate, migration_interval):
    # parameters used in DEAP framework
    pop = toolbox.population(n=population_size // num_islands)
    hof = tools.HallOfFame(1)

    # statistics for each island
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(f[0] for f in x) / len(x))
    stats.register("min", lambda x: min(f[0] for f in x))
    stats.register("max", lambda x: max(f[0] for f in x))

    # parameters for stagnation, cxpb, mutpb control
    best_fitness = None
    previous_best_fitness = float('inf')
    stagnation_counter = 0

    # main alg loop that runs for all the generations
    for gen in range(ngen):
        if gen == 0:
            cxpb = cxpb
            mutpb = mutpb
        else:
            cxpb = dynamic_cxpb_adaptive(previous_best_fitness, best_fitness, cxpb)
            mutpb = dynamic_mutpb_adaptive(previous_best_fitness, best_fitness, mutpb)

        if cxpb + mutpb > 1.0: # Normalize probabilities if their sum exceeds 1.0
            total = cxpb + mutpb
            cxpb /= total
            mutpb /= total

        # run the (mu, lambda) algorithm provided by DEAP
        pop, log = algorithms.eaMuCommaLambda(pop, toolbox, mu=population_size, lambda_=LAMBDA, cxpb=cxpb, mutpb=mutpb, ngen=1,
                                              stats=stats, halloffame=hof, verbose=False)

        # termination check
        current_best_fitness = hof[0].fitness.values[0]
        if current_best_fitness == 0:
            return pop, hof

        # Stagnation check: if fitness is the same or worse, increase counter
        if best_fitness is None or current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_threshold:
            increase_diversity(pop, toolbox, diversity_rate)
            stagnation_counter = 0

        # Ensure all individuals have their fitness values calculated
            # sometimes issues due to multiprocessing
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # Extract statistics for the current generation
        record = stats.compile(pop)
        if gen % migration_interval == 0:
            print(f"Island {island_id} - Generation: {gen}, avg: {record['avg']}, min: {record['min']}, max: {record['max']}")
        previous_best_fitness = current_best_fitness

    return pop, hof

# Main: called by Ass1Python.py and gridSearch.py. All parameters can be changed, or left on default (default = tuned parameter values)
def main(cxpb = 0.5, mutpb=0.1, block_size=4, tournament_size=3, lambda_multiplier = 2, stagnation_threshold=5, diversity_rate=0.4, num_islands=6, migration_rate=0.1, migration_interval=10, population_size = 1000, num_generation = 100):
    print(f"Running with population size: {population_size} and number of generations: {num_generation}")

    LAMBDA = population_size * lambda_multiplier  # Number of children to produce at each generation
    toolbox = set_deap_framework(block_size, tournament_size, mutpb)

    # Multiprocessing to run island algorithm
    with Pool(num_islands) as pool:
        results = [pool.apply_async(run_island, (toolbox, population_size, num_generation, LAMBDA, cxpb, mutpb, i, num_islands, stagnation_threshold, diversity_rate, migration_interval)) for i in range(num_islands)]
        populations_hofs = [result.get() for result in results]
        populations = [pop for pop, hof in populations_hofs]
        hofs = [hof for pop, hof in populations_hofs]

        for gen in range(num_generation):
            if gen % migration_interval == 0:
                migrate(populations, migration_rate)

    best_individuals = [hof[0] for hof in hofs]
    best_individual = min(best_individuals, key=lambda ind: ind.fitness.values)
    print(f"Best Fitness Value: {best_individual.fitness.values}")
    print(f"Best Puzzle: {best_individual}")
    write_output_file(best_individual)
    return best_individual.fitness.values[0]


