# Instructions 
- solve an 8x8 square puzzle containing 64 square pieces
- each tile has 4 sides and represented by 4 numbers
  - [top, right, bottom, left]
- there are 7 different motifs for the sides
- must read an input file with the 64 pieces then initialize the population changing positions and orientations of the pieces
- must output a file with the final puzzle
- create a command line UI to input pop size (in[100, 1000]) and number of generations (in[1,100])
- must test in c++ or java script to see the number of mismatching edges
  - 100% = 0-15 mismatched edges
- must run efficiently

# Structure

| Step                  | Functions                  | Strategy                                                                 | Input Parameters          |
|-----------------------|----------------------------|--------------------------------------------------------------------------|---------------------------|
| **Initialization**    | Read_input_file <br> Set_initial_puzzle <br> Rotate_piece <br> Toolbox.population          | Randomly shuffle and rotate the 64 pieces in Ass1Input.txt               | INPUT_FILENAME <br> POPULATION_SIZE <br> Pieces_list <br> Piece            |
| **Parent Selection**  | Tools.setTournament        | Running tournaments amongst a random subpopulation and the best individual from each tournament is a parent | TOURNAMENT_SIZE           |
| **Recombination**     | Block_crossover <br> Dynamic_cxpb_adaptive            | Selects a random block within puzzle and swaps it between parents. <br> Probability of crossover depends on current fitness (worse or equal fitness increases cxpb)      | Parent1 <br> Parent2 <br> BLOCK_SIZE <br> Prev_fitness <br> Current_fitness  <cxpb>                    |
| **Mutation**          | Mutate_puzzle <br> Dynamic_mutpb_adaptive             | Randomly replaces pieces in puzzle with random, rotated piece from original pieces <br> Probability of mutation depends on current fitness (worse or equal fitness increases mutpb)| Puzzle <br> Prev_fitness <br> Current_fitness <br> mutpb                   |
| **Survivor Selection**| algorithms.eaMuCommaLambda | (mu, lambda) selection: choose best of children pool                     | population_size <br> LAMBDA <br> num_generation                       |
| **Termination Conditions** | Main                  | If best fitness is 0 (minimization problem)  <br> Loop until number of generations is reached                             |                       |
| **Stagnation mitigation** | Increase_diversity     | If population becomes stagnant, replace some random puzzles with new puzzles created from original pieces | Population <br> Toolbox <br> Diversity_rate <br> STAGNATION_THRESHOLD             |




