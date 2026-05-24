from random import seed
from testing.utils import generate_data
from src.eoh.problems import Probs


prob = Probs("dvrptw_construct")
# problem_size = [10, 20, 50, 100, 200, 500]
problem_size = [10, 20, 50, 100, 200]
n_test = 64

if __name__ == "__main__":
    seed(2024)
    generate_data(prob, problem_size, n_test, n_thread=22)
