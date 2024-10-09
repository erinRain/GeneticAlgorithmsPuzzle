import island
from itertools import product

def grid_search():
    # Define the parameter grid
    cxpb_values = [0.5, 0.9]
    mutpb_values = [0.1, 0.3]
    block_size_values = [4, 5]
    tournament_size_values = [3]
    stagnation_threshold_values = [5]
    diversity_rate_values = [0.2, 0.4, 0.6]
    num_islands_values = [4, 6]
    migration_rate_values = [0.1, 0.2, 0.3]
    migration_interval_values = [10]

    # Initialize the best parameters and best score
    best_params = None
    # best_score = float('-inf')
    best_score = 102

    # Iterate over all combinations of parameters
    for params in product(cxpb_values, mutpb_values, block_size_values, tournament_size_values, stagnation_threshold_values, diversity_rate_values, num_islands_values, migration_rate_values, migration_interval_values):
        cxpb, mutpb, block_size, tournament_size, stagnation_threshold, diversity_rate, num_islands, migration_rate, migration_interval = params

        # Print the current parameters
        print(
            f"Running with parameters: cxpb={cxpb}, mutpb={mutpb}, block_size={block_size}, tournament_size={tournament_size}, stagnation_threshold={stagnation_threshold}, diversity_rate={diversity_rate}, num_islands={num_islands}, migration_rate={migration_rate}, migration_interval={migration_interval}")

        # Run the island.main function 3 times and calculate the average score
        total_score = 0
        for _ in range(3):
            score = island.main(cxpb, mutpb, block_size, tournament_size, stagnation_threshold, diversity_rate, num_islands, migration_rate, migration_interval)
            total_score += score
        average_score = total_score / 3

        # Update the best parameters and best score if the current score is better
        if average_score  < best_score:
            best_score = average_score
            best_params = params

    # Print the best parameters and best score
    print(
        f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")


if __name__ == "__main__":
    grid_search()
