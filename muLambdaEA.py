import random
from deap import base, creator, tools, algorithms
import fitness
from typing import List, Tuple
import time


## parameters
INPUT_FILENAME = "Ass1Input.txt"
OUTPUT_FILENAME = "OutputFiles/Ass1Output.txt"
POPULATION_SIZE = 1000
NGEN = 100  # Number of generations
LAMBDA = POPULATION_SIZE  # Number of children to produce at each generation
MU = LAMBDA//2  # Number of individuals to select for the next generation

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
def block_crossover(parent1, parent2):
    row_size = 8
    col_size = 8
    start_row = random.randint(0, row_size - BLOCK_SIZE)
    start_col = random.randint(0, col_size - BLOCK_SIZE)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            idx = (start_row + i) * col_size + (start_col + j)
            parent1[idx], parent2[idx] = parent2[idx], parent1[idx]
    return parent1, parent2

def dynamic_cxpb_adaptive(prev_fitness, current_fitness, cxpb):
    if current_fitness < prev_fitness:
        return max(0.1, cxpb * 0.9)
    else:
        return min(1.0, cxpb * 1.1)

# mutation: randomly replaces pieces in puzzle with a given probability MUTPB
def mutate_puzzle(puzzle: List[str], mutpb) -> tuple[list[str]]:
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

# calculate fitness of a puzzle
def calculate_fitness(puzzle: List[str]) -> Tuple[int]:
    return (fitness.main(puzzle),)

# prevent stagnation, replace some puzzles with new puzzles
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
def set_deap_framework():
    global pieces
    pieces = read_input_file()

    ## Set up type
    if hasattr(creator, 'FitnessMin'):
        del creator.FitnessMin
    if hasattr(creator, 'Individual'):
        del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    ## Initialization
    global toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_puzzle", set_initial_puzzle, pieces)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_puzzle)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    ## Operators
    toolbox.register("evaluate", calculate_fitness)
    toolbox.register("mate", block_crossover)
    toolbox.register("mutate", mutate_puzzle, mutpb = MUTPB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

def main(cxpb = 0.9, mutpb = 0.1, block_size = 4, tournament_size = 5, stagnation_threshold = 5, diversity_rate = 0.5):
    ## hyperparameters
    global CXPB, MUTPB, BLOCK_SIZE, TOURNAMENT_SIZE, STAGNATION_THRESHOLD, DIVERSITY_RATE
    CXPB = cxpb # Crossover probability
    MUTPB = mutpb  # Mutation probability
    BLOCK_SIZE = block_size  # crossover submatrix
    TOURNAMENT_SIZE = tournament_size  # survivor selection
    STAGNATION_THRESHOLD = stagnation_threshold  # Number of generations to wait before increasing diversity
    DIVERSITY_RATE = diversity_rate # Proportion of population to replace
    set_deap_framework()

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
        if current_best_fitness == 0: # termination
            break

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
        print(f"Generation: {gen}, avg: {record['avg']}, min: {record['min']}, max: {record['max']}")
        previous_best_fitness = current_best_fitness

    best_puzzle = hof[0]
    print(f"Best Fitness Value: {best_puzzle.fitness.values[0]}")
    print(f"Best Puzzle: {best_puzzle}")
    write_output_file(best_puzzle)
    return best_puzzle.fitness.values[0]

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Time: %s" % (time.time() - start_time))
