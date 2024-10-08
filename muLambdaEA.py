import random
from deap import base, creator, tools, algorithms
import fitness
from typing import List, Tuple
import math
import time

## parameters
INPUT_FILENAME = "Ass1Input.txt"
OUTPUT_FILENAME = "OutputFiles/Ass1Output.txt"
## hyperparameters
POPULATION_SIZE = 1000
NGEN = 100  # Number of generations
CXPB = 0.7 # Crossover probability
MUTPB = 0.3  # Mutation probability
BLOCK_SIZE = 4 # crossover submatrix
TOURNAMENT_SIZE = 3 # survivor selection
MU = 500  # Number of individuals to select for the next generation
LAMBDA = 1000  # Number of children to produce at each generation
STAGNATION_THRESHOLD = 5  # Number of generations to wait before increasing diversity
DIVERSITY_RATE = 0.3  # Proportion of population to replace

# read input file and create an array of the pieces
def read_input_file() -> List[str]:
    pieces_list = []
    with open(INPUT_FILENAME) as file:
        for line in file.readlines():
            piece = line.split(' ')
            for column in piece:
                pieces_list.append(column.strip())
    return pieces_list

# rotation: keep order and just change starting point (preserves sides)
def rotate_piece(piece: str) -> str:
    start = random.randint(0, len(piece) - 1)
    rotated = ''.join(piece[(start + i) % len(piece)] for i in range(len(piece)))
    return rotated

# set an initial puzzle with random positions and orientations
def set_initial_puzzle(pieces_list: List[str]) -> List[str]:
    random.shuffle(pieces_list)
    return [rotate_piece(piece) for piece in pieces_list]

# crossover: selects a random block within puzzle and swaps it between parents
    # aim: maintain local structures within puzzle so that matched sides are more likely to be kept together
    # probability of crossover is set in the eaMuCommaLambda algorithm
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

# mutation: randomly replaces pieces in puzzle with a given probability MUTPB
    # selects a random piece from original pieces in Ass1Input.txt and rotates it before placing it in puzzle
    # probability of mutation is set in the eaMuCommaLambda algorithm
def mutate_puzzle(puzzle: List[str]) -> tuple[list[str]]:
    for i in range(len(puzzle)):
        new_piece = random.choice(pieces)
        rotated_piece = rotate_piece(new_piece)
        puzzle[i] = rotated_piece
    return puzzle,

def dynamic_cxpb(gen, max_gen):
    return max(0.1, CXPB * (1 - gen / max_gen))

def dynamic_mutpb(gen, max_gen):
    return max(0.01, MUTPB * (1 - gen / max_gen))

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



# calculate fitness of a puzzle
def calculate_fitness(puzzle: List[str]) -> Tuple[int]:
    return (fitness.main(puzzle),)

def increase_diversity(population, toolbox, rate):
    num_to_replace = int(len(population) * rate)
    new_individuals = [toolbox.individual() for _ in range(num_to_replace)]
    population[-num_to_replace:] = new_individuals

# output
def write_output_file(puzzle):
    with open(OUTPUT_FILENAME, 'w') as file:
        file.write('Erin Rainville - 4019308')
        for i in range(0, len(puzzle), 8):
            row = ' '.join(puzzle[i:i+8])
            file.write('\n' + row)

# DEAP setup, get all puzzle pieces available
pieces = read_input_file()

## Set up type
# creates a FitnessMin class for a minimization problem (aim for least amount of mismatch)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creates an Individual class that is a simple list containing ints and has a fitness attribute FitnessMin
creator.create("Individual", list, fitness=creator.FitnessMin)

## Initialization
toolbox = base.Toolbox()
# set up function attr_puzzle by calling set_initial_puzzle(pieces)
    # reminder: pieces is all original pieces in Ass1Input.txt file
toolbox.register("attr_puzzle", set_initial_puzzle, pieces)
# set up function to create new individuals by calling attr_puzzle function to generate initial puzzle and wrapping it in an Individual object
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_puzzle)
# set up function to initialize a population as a list of puzzles by calling initRepeat which will call individual for n times
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

## Operators
    # must return tuples because single-objective is a special case of multi-objective
# set up evaluation function which calculates the fitness of a puzzle
toolbox.register("evaluate", calculate_fitness)
# set up crossover function which randomly swaps a submatrix of two parents
toolbox.register("mate", block_crossover)
# set up mutation function which replaces random pieces in a puzzle with a randomly rotated, random piece from Ass1Input pieces
toolbox.register("mutate", mutate_puzzle)
# set up selection function
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

def main():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(f[0] for f in x) / len(x))
    stats.register("min", lambda x: min(f[0] for f in x))
    stats.register("max", lambda x: max(f[0] for f in x))

    best_fitness = None
    previous_best_fitness = float('inf')
    stagnation_counter = 0

    for gen in range(NGEN):
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
            print("fix stagnation")

        # Ensure all individuals have their fitness values calculated
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # Extract statistics for the current generation
        record = stats.compile(pop)
        print(f"Generation: {gen}, avg: {record['avg']}, min: {record['min']}, max: {record['max']}")
        previous_best_fitness = current_best_fitness

    best_puzzle = hof[0]
    print(f"Best Fitness Value: {best_puzzle.fitness.values[0]}")
    print(f"Best Puzzle: {best_puzzle}")
    write_output_file(best_puzzle)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Time: %s" % (time.time() - start_time))
