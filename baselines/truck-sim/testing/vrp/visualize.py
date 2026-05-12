import pickle
import random
import importlib

import cProfile

import matplotlib.pyplot as plt

from testing.utils import plot_solution
from src.eoh.problems.vrp_construct.run import VRPConstruct
from src.eoh.problems.vrp_gls.run import VRPGLS


to_test = []
# problem_size = [10, 20, 50, 100, 200]
problem_size = [50, 100, 200]

random.seed(1234)
for size in problem_size:
    instance_file_name = f"./testing_data/instance_data_{size}.pkl"
    with open(instance_file_name, 'rb') as f:
        instance_dataset = pickle.load(f)

    instance = random.choice(instance_dataset)
    to_test.append((instance, size))


# with open("./training_data/instances.pkl", "rb") as f:
#     instance_dataset = pickle.load(f)
# for instance in instance_dataset:
#     to_test.append((instance, 50))

if __name__ == '__main__':
    # heuristic_module = importlib.import_module("vrp_construct.heuristic")
    # alg = importlib.reload(heuristic_module)
    #
    # heuristic_module = importlib.import_module("vrp_construct.greedy")
    # gre = importlib.reload(heuristic_module)

    heuristic_module = importlib.import_module("vrp_gls.heuristic")
    alg = importlib.reload(heuristic_module)

    # heuristic_module = importlib.import_module("vrp_gls.identity")
    # gre = importlib.reload(heuristic_module)

    for instance, size in to_test:
        print(f"\nDoing size {size}...")
        # eva = VRPConstruct([instance], size, 1)
        eva = VRPGLS([instance], size, 1)

        cProfile.run('eva.eval(alg)', sort="cumtime")

        # greedy_score, greedy_results = eva.eval(gre)
        score, results = eva.eval(alg)
        coords, distances, ortool = instance

        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 8))
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
        #
        # ax1.title.set_text(f"Greedy (Score: {greedy_score})")
        # plot_solution(coords, greedy_results["routes"][0], ax1)

        ax2.title.set_text(f"Heuristic (Score: {score})")
        plot_solution(coords, results["routes"][0], ax2)

        ax3.title.set_text(f"ORTool")
        plot_solution(coords, ortool["routes"], ax3)

        fig.show()
