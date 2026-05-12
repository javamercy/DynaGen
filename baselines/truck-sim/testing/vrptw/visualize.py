import time
import pickle
import random
import importlib

import numpy as np
import matplotlib.pyplot as plt

from src.eoh.problems.vrptw_construct import VRPTWConstruct

from testing.utils import plot_solution


if __name__ == '__main__':
    heuristic_module = importlib.import_module("vrptw_construct.heuristic")
    alg = importlib.reload(heuristic_module)

    heuristic_module = importlib.import_module("vrptw_construct.greedy")
    gre = importlib.reload(heuristic_module)

    problem_size = [10, 20, 50, 100, 200]
    # problem_size = [100]

    random.seed(1234)
    # random.seed(123)
    for size in problem_size:
        instance_file_name = f"./testing_data/instance_data_{size}.pkl"
        with open(instance_file_name, 'rb') as f:
            instance_dataset = pickle.load(f)

        instance = random.choice(instance_dataset)
        coords = instance[0]

        eva = VRPTWConstruct([instance], size, 1)

        greedy_score, greedy_results = eva.eval(gre)
        score, results = eva.eval(alg)

        g_routes = greedy_results["routes"][0]
        routes = results["routes"][0]

        _, distances, ortool = instance

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

        ax1.title.set_text(f"Greedy (Score: {greedy_score})")
        plot_solution(coords, g_routes, ax1)

        ax2.title.set_text(f"Heuristic (Score: {score})")
        plot_solution(coords, routes, ax2)

        ax3.title.set_text(f"ORTool")
        plot_solution(coords, ortool["routes"], ax3)

        fig.show()
