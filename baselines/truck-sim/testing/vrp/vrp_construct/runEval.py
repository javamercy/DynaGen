import os
import time
import pickle
import importlib
import numpy as np

from src.eoh.problems.vrp_construct.run import VRPConstruct


if __name__ == '__main__':
    heuristic_module = importlib.import_module("heuristic")
    alg = importlib.reload(heuristic_module)

    heuristic_module = importlib.import_module("greedy")
    gre = importlib.reload(heuristic_module)

    problem_size = [10, 20, 50, 100, 200]
    n_test_ins = 64
    print("Start evaluation...")

    times = []
    ortool = []
    greedy = []
    g_score = []
    heuristic = []
    h_score = []

    filename = "results/results.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:

        for size in problem_size:
            instance_file_name = f"../testing_data/instance_data_{size}.pkl"
            with open(instance_file_name, 'rb') as f:
                instance_dataset = pickle.load(f)

            eva = VRPConstruct(instance_dataset, size, n_test_ins)

            greedy_score, greedy_results = eva.eval(gre)

            time_start = time.time()
            score, results = eva.eval(alg)

            time_res = time.time() - time_start
            # ortool_res = np.average(results['ortool'])
            ortool_res = np.average([ort['max_distance'] for ort in results['ortool']])
            greedy_res = np.average(greedy_results['heuristic'])
            g_score_res = np.average(greedy_score)
            heuristic_res = np.average(results['heuristic'])
            h_score_res = score

            result = (
                f"Instance size {size}:\n"
                f" - Time taken: {time_res:.3f}\n"
                f" - OR-Tool avg time: {ortool_res:.3f}\n"
                f" - Greedy avg time: {greedy_res:.3f}\n"
                f" - Greedy Score: {g_score_res:.3f}\n"
                f" - Heuristic avg time: {heuristic_res:.3f}\n"
                f" - Heuristic Score: {h_score_res:.3f}\n"
            )
            print(result)
            file.write(result + "\n")

            times.append(time_res)
            ortool.append(ortool_res)
            greedy.append(greedy_res)
            g_score.append(g_score_res)
            heuristic.append(heuristic_res)
            h_score.append(h_score_res)

    print("Time (Sec)" + "".join(f" & {x:.3f}" for x in times) + " \\\\")
    print("OR-Tool" + "".join(f" & {x:.3f}" for x in ortool) + " \\\\")
    print("Greedy" + "".join(f" & {x:.3f}" for x in greedy) + " \\\\")
    print("Score" + "".join(f" & {x:.3f}" for x in g_score) + " \\\\")
    print("Heuristic" + "".join(f" & {x:.3f}" for x in heuristic) + " \\\\")
    print("Score" + "".join(f" & {x:.3f}" for x in h_score) + " \\\\")
