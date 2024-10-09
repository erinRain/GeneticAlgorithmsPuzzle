import random
from deap import base, creator, tools, algorithms
import fitness
from typing import List, Tuple
from multiprocessing import Pool
import time

# Unchanging Parameters
INPUT_FILENAME = "Ass1Input.txt"
OUTPUT_FILENAME = "OutputFiles/Ass1Output.txt"
POPULATION_SIZE = 1000
NGEN = 100  # Number of generations
LAMBDA = POPULATION_SIZE  # Number of children to produce at each generation
MU = LAMBDA//2  # Number of individuals to select for the next generation

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
def block_crossover(parent1, parent2, block_size):
    row_size = 8
    col_size = 8
    if block_size > row_size or block_size > col_size:
        raise ValueError(f"block_size {block_size}must be less than or equal to row_size and col_size")
    start_row = random.randint(0, row_size - block_size)
    start_col = random.randint(0, col_size - block_size)
    for i in range(block_size):
        for j in range(block_size):
            idx = (start_row + i) * col_size + (start_col + j)
            parent1[idx], parent2[idx] = parent2[idx], parent1[idx]
    return parent1, parent2

def dynamic_cxpb_adaptive(prev_fitness, current_fitness, cxpb):
    if current_fitness < prev_fitness:
        return max(0.1, cxpb * 0.9)
    else:
        return min(1.0, cxpb * 1.1)

# Mutation: randomly replaces pieces in puzzle with a given probability MUTPB
def mutate_puzzle(puzzle: List[str], pieces, mutpb) -> tuple[list[str]]:
    for i in range(len(puzzle)):
        if random.random() < mutpb:
            new_piece = random.choice(pieces)
            rotated_piece = rotate_piece(new_piece)
            puzzle[i] = rotated_piece
    return puzzle,

def dynamic_mutpb_adaptive(prev_fitness, current_fitness, mutpb):
    if current_fitness < prev_fitness:
        return max(0.01, mutpb * 0.9)
    else:
        return min(1.0, mutpb * 1.1)

# Calculate fitness of a puzzle
def calculate_fitness(puzzle: List[str]) -> Tuple[int]:
    return (fitness.main(puzzle),)

# prevent stagnation, replace some puzzles with new puzzles
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
def set_deap_framework(block_size, tournament_size, mutpb):
    pieces = read_input_file()

    # Set up type
    if hasattr(creator, 'FitnessMin'):
        del creator.FitnessMin
    if hasattr(creator, 'Individual'):
        del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialization
    toolbox = base.Toolbox()
    toolbox.register("attr_puzzle", set_initial_puzzle, pieces)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_puzzle)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operators
    toolbox.register("evaluate", calculate_fitness)
    toolbox.register("mate", block_crossover, block_size = block_size)
    toolbox.register("mutate", mutate_puzzle, pieces = pieces, mutpb = mutpb)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    return toolbox

def migrate(populations, migration_rate):
    num_migrants = int(len(populations) * migration_rate)
    for i in range(len(populations)):
        next_island = (i + 1) % len(populations)
        migrants = random.sample(populations[i], num_migrants)
        populations[next_island].extend(migrants)
        populations[i] = [ind for ind in populations[i] if ind not in migrants]

def run_island(toolbox, ngen, cxpb, mutpb, island_id, num_islands, stagnation_threshold, diversity_rate, migration_interval):
    pop = toolbox.population(n=POPULATION_SIZE // num_islands)
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
            cxpb = cxpb
            mutpb = mutpb
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
        if current_best_fitness == 0: # termination
            break

        if best_fitness is None or current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_threshold:
            increase_diversity(pop, toolbox, diversity_rate)
            stagnation_counter = 0

        # Ensure all individuals have their fitness values calculated
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # Extract statistics for the current generation
        record = stats.compile(pop)
        #if gen % migration_interval == 0:
        #    print(f"Island {island_id} - Generation: {gen}, avg: {record['avg']}, min: {record['min']}, max: {record['max']}")
        previous_best_fitness = current_best_fitness

    return pop, hof

def main(cxpb = 0.5, mutpb=0.1, block_size=4, tournament_size=3, stagnation_threshold=5, diversity_rate=0.4, num_islands=6, migration_rate=0.1, migration_interval=10):

    toolbox = set_deap_framework(block_size, tournament_size, mutpb)

    with Pool(num_islands) as pool:
        results = [pool.apply_async(run_island, (toolbox, NGEN, cxpb, mutpb, i, num_islands, stagnation_threshold, diversity_rate, migration_interval)) for i in range(num_islands)]
        populations_hofs = [result.get() for result in results]
        populations = [pop for pop, hof in populations_hofs]
        hofs = [hof for pop, hof in populations_hofs]

        for gen in range(NGEN):
            if gen % migration_interval == 0:
                migrate(populations, migration_rate)

    best_individuals = [hof[0] for hof in hofs]
    best_individual = min(best_individuals, key=lambda ind: ind.fitness.values)
    print(f"Best Fitness Value: {best_individual.fitness.values}")
    print(f"Best Puzzle: {best_individual}")
    write_output_file(best_individual)
    return best_individual.fitness.values[0]

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Time: %s" % (time.time() - start_time))
