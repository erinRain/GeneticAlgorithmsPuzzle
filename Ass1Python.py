## Erin Rainville - 40179308

import random
import subprocess

from IPython.sphinxext.ipython_directive import OUTPUT

## parameters
INPUT_FILENAME = "Ass1Input.txt"
OUTPUT_FILENAME = "C:/Users/Erin/Documents/COEN6321 Machine Learning/OutputFiles/Ass1Output.txt"
TESTER_FILENAME = "C:/Users/Erin/Documents/COEN6321 Machine Learning/tester/x64/Debug/tester.exe"
## hyperparameters

# read input file and create an array of the pieces
def read_input_file():
    pieces_list = []
    file = open(INPUT_FILENAME)
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
def fitness_validation():
    result = subprocess.run([TESTER_FILENAME], capture_output=True, text=True )
    output = result.stdout
    return int(output.split(":")[1].strip())


# output
def write_output_file(population):
    with open(OUTPUT_FILENAME, 'w') as file:
        file.write('Erin Rainville - 4019308 \n')
        for i in range(0, len(population), 8):
            row = ' '.join(population[i:i+8])
            file.write(row + '\n')



if __name__ == "__main__":
    pieces = read_input_file()
    population = set_initial_population(pieces)
    write_output_file(population)
    print(population)
    fitness_validation()
