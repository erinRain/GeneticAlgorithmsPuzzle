# Quick Setup
- download the project
- in the project's terminal (population_size can be anything in [100,1000], num_generation [1,100]): 
```
python island.py --population_size 1000 --num_generation 100
```

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

# Algorithm
1. Initialization:
   - Set population size and number of generations in command prompt, repeats until values are in correct range
   - Read Input: Reads puzzle pieces from a file.
   - Rotate Pieces: Randomly rotates each piece.
   - Set Initial Puzzle: Shuffles and rotates pieces to create an initial puzzle configuration.
2. Genetic Operators:
   - Crossover: Swaps blocks of pieces between two parent puzzles. (Blocks to keep potential matches together).
   - Mutation: Randomly replaces pieces in the puzzle with a given probability.
3. Adaptive Parameters:
   - Crossover Probability (cxpb): Adjusts based on fitness improvement.
   - Mutation Probability (mutpb): Adjusts similarly to crossover probability.
4. Fitness Calculation:
   - Calculate Fitness: Evaluates how well the puzzle is solved, using given fitness file, minimizing problem.
5. Diversity and Stagnation:
   - Increase Diversity: Replaces some puzzles with new ones to prevent stagnation.
   - Stagnation Check: Monitors fitness improvement to trigger diversity increase.
6. Island Model:
   - Multiple Islands: Runs multiple sub-populations (islands) in parallel.
   - Migration: Periodically exchanges individuals between islands to maintain diversity.
7. Main Evolution Loop:
   - Run Generations: Evolves the population over a specified number of generations.
   - Adaptive Parameters: Continuously adjusts crossover and mutation probabilities.
   - Migration: Ensures periodic migration between islands.
8. Output:
   - Write Output: Writes the best puzzle configuration to a file.
9. Framework
   - the prorgam uses the DEAP framework with the eaMuCommaLambda algorithm to simplify the EA programming


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

# Results
After pamater tuning, the final inputs with population_size 1000 and num_generation 100 are:

cxpb = 0.5, mutpb=0.1, block_size=4, tournament_size=3, lambda_multiplier = 2, stagnation_threshold=5, diversity_rate=0.4, num_islands=6, migration_rate=0.1, migration_interval

It produces an average fitness of 16 mismatches in ~22 seconds

If the population size is increased to 10,000 with mu 10

A fitness of 0 mismatches was accomplished:
```
2115 0261 1042 1210 4442 5264 2512 1415
1120 6401 4244 1412 4514 6145 1201 1042
2115 0221 4212 1412 1204 4212 0112 4411
1026 2610 1116 1201 0112 1321 1023 1120
2005 1120 1201 0162 1441 2124 2101 2141
0121 2261 0112 6111 4151 2261 0412 4614
2102 6231 1412 1144 5141 6521 1605 1116
0423 3134 1161 4151 4121 2421 0514 1415
```





