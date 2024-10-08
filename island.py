import random
from deap import base, creator, tools, algorithms
import fitness
from typing import List, Tuple
import math
from multiprocessing import Pool
import time


# Unchanging Parameters
INPUT_FILENAME = "Ass1Input.txt"
OUTPUT_FILENAME = "OutputFiles/Ass1Output.txt"
POPULATION_SIZE = 1000
NGEN = 100
MU = 500
LAMBDA = 1000

# HyperParameteres
CXPB = 0.7
MUTPB = 0.3
BLOCK_SIZE = 4
TOURNAMENT_SIZE = 3
STAGNATION_THRESHOLD = 5
DIVERSITY_RATE = 0.3
NUM_ISLANDS = 4
MIGRATION_RATE = 0.1
MIGRATION_INTERVAL = 10

# Read input file and create an array of the pieces
def read_input_file() -> List[str]:
    pieces_list = []
    with open(INPUT_FILENAME) as file:
        for line in file.readlines():
            piece = line.split(' ')
            for column in piece:
                pieces_list.append(column.strip())
    return pieces_list

# Rotation: keep order and just change starting point (preserves sides)
def rotate_piece(piece: str) -> str:
    start = random.randint(0, len(piece) - 1)
    rotated = ''.join(piece[(start + i) % len(piece)] for i in range(len(piece)))
    return rotated

# Set an initial puzzle with random positions and orientations
def set_initial_puzzle(pieces_list: List[str]) -> List[str]:
    random.shuffle(pieces_list)
    return [rotate_piece(piece) for piece in pieces_list]

# Crossover: selects a random block within puzzle and swaps it between parents
def block_crossover(parent1, parent2):
    row_size = 8
    col_size = 8
    start_row = random.randint(0, row_size - BLOCK_SIZE)
    start_col = random.randint(0, col_size - BLOCK_SIZE)

    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            idx1 = (start_row + i) * col_size + (start_col + j)
            idx2 = (start_row + i) * col_size + (start_col + j)
            parent1[idx1], parent2[idx2] = parent2[idx2], parent1[idx1]
    return parent1, parent2

# Mutation: randomly replaces pieces in puzzle with a given probability MUTPB
def mutate_puzzle(puzzle: List[str]) -> tuple[list[str]]:
    for i in range(len(puzzle)):
        new_piece = random.choice(pieces)
        rotated_piece = rotate_piece(new_piece)
        puzzle[i] = rotated_piece
    return puzzle,

def dynamic_cxpb_adaptive(prev_fitness, current_fitness, cxpb):
    if current_fitness < prev_fitness:
        return max(0.1, cxpb * 0.9)
    else:
        return min(1.0, cxpb * 1.1)

def dynamic_mutpb_adaptive(prev_fitness, current_fitness, mutpb):
    if current_fitness < prev_fitness:
        return max(0.01, mutpb * 0.9)
    else:
        return min(1.0, mutpb * 1.1)

# Calculate fitness of a puzzle
def calculate_fitness(puzzle: List[str]) -> Tuple[int]:
    return (fitness.main(puzzle),)

def increase_diversity(population, toolbox, rate):
    num_to_replace = int(len(population) * rate)
    new_individuals = [toolbox.individual() for _ in range(num_to_replace)]
    population[-num_to_replace:] = new_individuals

# Output
def write_output_file(puzzle):
    with open(OUTPUT_FILENAME, 'w') as file:
        file.write('Erin Rainville - 4019308')
        for i in range(0, len(puzzle), 8):
            row = ' '.join(puzzle[i:i+8])
            file.write('\n' + row)

# DEAP setup, get all puzzle pieces available
pieces = read_input_file()

# Set up type
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialization
toolbox = base.Toolbox()
toolbox.register("attr_puzzle", set_initial_puzzle, pieces)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_puzzle)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operators
toolbox.register("evaluate", calculate_fitness)
toolbox.register("mate", block_crossover)
toolbox.register("mutate", mutate_puzzle)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

def migrate(populations, migration_rate):
    num_migrants = int(len(populations) * migration_rate)
    for i in range(len(populations)):
        next_island = (i + 1) % len(populations)
        migrants = random.sample(populations[i], num_migrants)
        populations[next_island][0].extend(migrants)
        populations[i] = [ind for ind in populations[i] if ind not in migrants]

def run_island(toolbox, ngen, cxpb, mutpb, island_id):
    pop = toolbox.population(n=POPULATION_SIZE // NUM_ISLANDS)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(f[0] for f in x) / len(x))
    stats.register("min", lambda x: min(f[0] for f in x))
    stats.register("max", lambda x: max(f[0] for f in x))

    best_fitness = None
    previous_best_fitness = float('inf')
    stagnation_counter = 0

    for gen in range(ngen):
        if gen == 0:
            cxpb = CXPB
            mutpb = MUTPB
        else:
            cxpb = dynamic_cxpb_adaptive(previous_best_fitness, best_fitness, cxpb)
            mutpb = dynamic_mutpb_adaptive(previous_best_fitness, best_fitness, mutpb)

        # Normalize probabilities if their sum exceeds 1.0
        if cxpb + mutpb > 1.0:
            total = cxpb + mutpb
            cxpb /= total
            mutpb /= total

        pop, log = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, cxpb=cxpb, mutpb=mutpb, ngen=1,
                                              stats=stats, halloffame=hof, verbose=False)

        current_best_fitness = hof[0].fitness.values[0]
        if best_fitness is None or current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= STAGNATION_THRESHOLD:
            increase_diversity(pop, toolbox, DIVERSITY_RATE)
            stagnation_counter = 0

        # Ensure all individuals have their fitness values calculated
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # Extract statistics for the current generation
        record = stats.compile(pop)
        previous_best_fitness = current_best_fitness

        if gen % MIGRATION_INTERVAL == 0:
            print(f"Island {island_id} - Generation: {gen}, avg: {record['avg']}, min: {record['min']}, max: {record['max']}")

    return pop, hof

def main():

    with Pool(NUM_ISLANDS) as pool:
        results = [pool.apply_async(run_island, (toolbox, NGEN, CXPB, MUTPB, i)) for i in range(NUM_ISLANDS)]
        populations_hofs = [result.get() for result in results]
        populations = [pop for pop, hof in populations_hofs]
        hofs = [hof for pop, hof in populations_hofs]

        for gen in range(NGEN):
            if gen % MIGRATION_INTERVAL == 0:
                migrate(populations, MIGRATION_RATE)

    best_individuals = [hof[0] for hof in hofs]
    best_individual = min(best_individuals, key=lambda ind: ind.fitness.values)
    print(f"Best Fitness Value: {best_individual.fitness.values}")
    print(f"Best Puzzle: {best_individual}")
    write_output_file(best_individual)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Time: %s" % (time.time() - start_time))
