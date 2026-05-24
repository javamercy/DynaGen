import time
import pickle
import random
import importlib

import numpy as np
import matplotlib.pyplot as plt

from src.eoh.problems.dvrp_construct.run import DVRPConstruct

from testing.utils import plot_solution


if __name__ == '__main__':
    heuristic_module = importlib.import_module("heuristic")
    alg = importlib.reload(heuristic_module)

    heuristic_module = importlib.import_module("greedy")
    gre = importlib.reload(heuristic_module)

    problem_size = [10, 20, 50, 100, 200]

    random.seed(1234)
    for size in problem_size:
        instance_file_name = f"./testing_data/instance_data_{size}.pkl"
        with open(instance_file_name, 'rb') as f:
            instance_dataset = pickle.load(f)

        instance = random.choice(instance_dataset)
        coords = instance[0]

        eva = DVRPConstruct([instance], size, 1)

        greedy_score, greedy_results = eva.eval(gre)
        score, results = eva.eval(alg)

        g_routes = greedy_results["routes"][0]
        routes = results["routes"][0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.title.set_text(f"Greedy (Score: {greedy_score})")
        plot_solution(coords, g_routes, ax1)

        ax2.title.set_text(f"Heuristic (Score: {score})")
        plot_solution(coords, routes, ax2)

        fig.show()


    #     result = (
    #         f"Instance size {size}:\n"
    #         f" - Time taken: {time_res:.3f}\n"
    #         f" - OR-Tool avg time: {ortool_res:.3f}\n"
    #         f" - Greedy avg time: {greedy_res:.3f}\n"
    #         f" - Greedy Score: {g_score_res:.3f}\n"
    #         f" - Heuristic avg time: {heuristic_res:.3f}\n"
    #         f" - Heuristic Score: {h_score_res:.3f}\n"
    #     )
    #     print(result)
    #     file.write(result + "\n")
    #
    #     times.append(time_res)
    #     ortool.append(ortool_res)
    #     greedy.append(greedy_res)
    #     g_score.append(g_score_res)
    #     heuristic.append(heuristic_res)
    #     h_score.append(h_score_res)
    #
    # print("Time (Sec)" + "".join(f" & {x:.3f}" for x in times) + " \\\\")
    # print("OR-Tool" + "".join(f" & {x:.3f}" for x in ortool) + " \\\\")
    # print("Greedy" + "".join(f" & {x:.3f}" for x in greedy) + " \\\\")
    # print("Score" + "".join(f" & {x:.3f}" for x in g_score) + " \\\\")
    # print("Heuristic" + "".join(f" & {x:.3f}" for x in heuristic) + " \\\\")
    # print("Score" + "".join(f" & {x:.3f}" for x in h_score) + " \\\\")
