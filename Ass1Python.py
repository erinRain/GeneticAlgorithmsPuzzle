# Erin Rainville - 40179308
# Command Prompt UI for Assignment - start of program

import subprocess
import sys

# the algorithm uses the DEAP framework, so it installs deap if it is not installed
# Ensures that anyone can run the program as long as they have python installed
def install_deap():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap"])

try:
    import deap
except ImportError:
    print("DEAP is not installed. Installing now...")
    install_deap()
    import deap
    print("DEAP has been installed successfully.")

# If the user put numbers that aren't within the range, will continue asking for a correct value
def get_valid_input(prompt, valid_range):
    while True:
        try:
            value = int(input(prompt))
            if value in valid_range:
                return value
            else:
                print(f"Value must be within the range {valid_range.start} to {valid_range.stop - 1}. Please try again.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# What is run when calling python Ass1Python.py
if __name__ == "__main__":
    import time
    import argparse
    import algorithm

    # setting population and generations values in command line
    parser = argparse.ArgumentParser(description="Evolutionary Algorithm Parameters")
    parser.add_argument("--population_size", type=int, help="Population size (in [100, 1000])")
    parser.add_argument("--num_generation", type=int, help="Number of generations (in [1, 100])")

    args = parser.parse_args()

    if args.population_size is None or not (100 <= args.population_size <= 1000):
        args.population_size = get_valid_input("Enter population size (in [100, 1000]): ", range(100, 1001))
    if args.num_generation is None or not (1 <= args.num_generation <= 100):
        args.num_generation = get_valid_input("Enter number of generations (in [1, 100]): ", range(1, 101))

    # Run the algorithm and calculate the time it took
    start_time = time.time()
    algorithm.main(population_size=args.population_size, num_generation=args.num_generation)
    print("Time taken to do algorithm: %s" % (time.time() - start_time))
    print(r"The output can be found in Ass1Python\OutputFiles\Ass1Output.txt")