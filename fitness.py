# Erin Rainville - 40179308
# fitness evaluation of mismatched row using the fitness.cpp file given in assignment instructions
    # changed it to python
 # input_data is a list of strings

def count_row_mismatches(first, second):
    number_of_mismatches = 0
    for i in range(len(first)):
        if first[i][2] != second[i][0]:
            number_of_mismatches += 1
    return number_of_mismatches

def count_column_mismatches(first, second):
    number_of_mismatches = 0
    for i in range(len(first)):
        if first[i][1] != second[i][3]:
            number_of_mismatches += 1
    return number_of_mismatches

def main(input_data):
    row_size = 8
    column_size = 8

    puzzle_pieces = []
    for i in range(row_size):
        row = input_data[i * column_size:(i + 1) * column_size]
        puzzle_pieces.append(row)

    number_of_mismatches = 0

    for i in range(row_size - 1):
        number_of_mismatches += count_row_mismatches(puzzle_pieces[i], puzzle_pieces[i + 1])

    for i in range(column_size - 1):
        first_column = [puzzle_pieces[j][i] for j in range(row_size)]
        second_column = [puzzle_pieces[j][i + 1] for j in range(row_size)]
        number_of_mismatches += count_column_mismatches(first_column, second_column)

    return number_of_mismatches

