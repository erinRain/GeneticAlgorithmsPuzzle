## Erin Rainville - 40179308

import csv
import random

## parameters
FILE_NAME = "Ass1Input.txt"
## hyperparameters

# read input file and create an array of the pieces
def read_input_file(filename):
    pieces_list = []
    file = open(filename)
    for line in file.readlines():
        fields = line.split(' ')
        for column in fields:
            pieces_list.append(column.strip())
    file.close()
    return pieces_list

# set the initial population with random positions and orientations
def set_initial_population(pieces_list):
    random.shuffle(pieces_list)
    print(pieces_list)
    return [rotate_pieces(piece) for piece in pieces_list]

# rotation: keep order and just change starting point
def rotate_pieces(piece):
    start = random.randint(0, len(piece) - 1)
    rotated = ''.join(piece[(start + i) %len(piece)] for i in range(len(piece)))
    return rotated

# fitness test

if __name__ == "__main__":
    pieces = read_input_file(FILE_NAME)
    #print(pieces)
    population = set_initial_population(pieces)
    print(population)
