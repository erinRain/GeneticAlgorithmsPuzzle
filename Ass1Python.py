## Erin Rainville - 40179308

import random
#import subprocess
import fitness
from typing import List, Tuple


## parameters
INPUT_FILENAME = "Ass1Input.txt"
OUTPUT_FILENAME = "OutputFiles/Ass1Output.txt"
## hyperparameters
POPULATION_SIZE = 100

# read input file and create an array of the pieces
def read_input_file() -> List[str]:
    pieces_list = []
    with open(INPUT_FILENAME) as file:
        for line in file.readlines():
            piece = line.split(' ')
            for column in piece:
                pieces_list.append(column.strip())
    return pieces_list

# initial population
    # population_size number of puzzles are generated
    # call the set_initial_puzzle which makes puzzle random pos and orientation
    # create the output file which check
def set_initial_population(pieces_list: List[str]) -> List[List[str]]:
    puzzles_list = []
    for i in range(POPULATION_SIZE):
        puzzles_list.append(set_initial_puzzle(pieces_list))
    return puzzles_list

# set an initial puzzle with random positions and orientations
def set_initial_puzzle(pieces_list: List[str]) -> List[str]:
    random.shuffle(pieces_list)
    return [rotate_pieces(piece) for piece in pieces_list]

# rotation: keep order and just change starting point
def rotate_pieces(piece: str) -> str:
    start = random.randint(0, len(piece) - 1)
    rotated = ''.join(piece[(start + i) % len(piece)] for i in range(len(piece)))
    return rotated

# calculate fitness of a puzzle
def calculate_fitness(puzzle: List[str]) -> int:
    return fitness.main(puzzle)

# get the winner of the fitness
def get_best_fitness(puzzles_list: List[List[str]]) -> Tuple[int, List[str]]:
    b_value = 102
    b_puzzle = None
    for puzzle in puzzles_list:
        result = calculate_fitness(puzzle)
        if result < b_value:
            b_value = result
            b_puzzle = puzzle
    return b_value, b_puzzle

# output
def write_output_file(puzzle):
    with open(OUTPUT_FILENAME, 'w') as file:
        file.write('Erin Rainville - 4019308')
        for i in range(0, len(puzzle), 8):
            row = ' '.join(puzzle[i:i+8])
            file.write('\n' + row)



if __name__ == "__main__":
    pieces = read_input_file()
    #print(pieces)
    puzzles = set_initial_population(pieces)
    best_value, best_puzzle = get_best_fitness(puzzles)
    print(f"Best Fitness Value: {best_value}")
    print(f"Best Puzzle: {best_puzzle}")
    write_output_file(best_puzzle)

