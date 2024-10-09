import itertools
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import muLambdaEA
import threading


# Flag to indicate interruption
interrupted = threading.Event()

def grid_search_muLambdaEa():
    # Define the parameter grid
    cxpb_values = [0.9]
    mutpb_values = [0.1]
    block_size_values = [4]
    tournament_size_values = [5]
    stagnation_threshold_values = [5, 10]
    diversity_rate_values = [0.2, 0.4]

    # Initialize the best parameters and best score
    best_params = None
    best_score = float(120.0)

    def signal_handler(sig, frame):
        print("\nProcess interrupted!")
        interrupted.set()
        if best_params:
            print(f"Best parameters so far: {best_params}")
            print(f"Best score so far: {best_score}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    def evaluate_params(params):
        if interrupted.is_set():
            return float('inf'), params  # Return a high score if interrupted
        cxpb, mutpb, block_size, tournament_size, stagnation_threshold, diversity_rate = params
        scores = []
        print(
            f"Running with parameters: cxpb={cxpb}, mutpb={mutpb}, block_size={block_size}, tournament_size={tournament_size}, stagnation_threshold={stagnation_threshold},diversity_rate={diversity_rate} \n")

        for _ in range(3):
            if interrupted.is_set():
                return float('inf'), params  # Return a high score if interrupted
            score = muLambdaEA.main(cxpb, mutpb, block_size, tournament_size, stagnation_threshold, diversity_rate)
            scores.append(score)
        avg_score = sum(scores) / len(scores)
        return avg_score, params

    # Create a thread pool executor
    with ThreadPoolExecutor() as executor:
        futures = []
        for params in itertools.product(cxpb_values, mutpb_values, block_size_values, tournament_size_values, stagnation_threshold_values, diversity_rate_values):
            futures.append(executor.submit(evaluate_params, params))

        for future in as_completed(futures):
            avg_score, params = future.result()
            if avg_score < best_score:
                best_score = avg_score
                best_params = params

    # Print the best parameters and best score
    if best_params:
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")

if __name__ == "__main__":
    grid_search_muLambdaEa()
