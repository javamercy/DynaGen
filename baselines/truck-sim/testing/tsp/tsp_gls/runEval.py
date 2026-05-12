import os
import time
import pickle
import importlib
import numpy as np
import matplotlib.pyplot as plt

from src.eoh.problems.tsp_gls.run import TSPGLS


if __name__ == "__main__":
    heuristic_module = importlib.import_module("identity")
    ide = importlib.reload(heuristic_module)

    heuristic_module = importlib.import_module("heuristic")
    alg = importlib.reload(heuristic_module)

    problem_size = [10, 20, 50, 100, 200]
    n_test_ins = 64
    print("Start evaluation...")

    times = []
    ortool = []
    identi = []
    i_score = []
    heuristic = []
    h_score = []

    plot_i_score = []
    plot_h_score = []

    filename = "./results/results.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        for size in problem_size:
            instance_file_name = f"../testing_data/instance_data_{size}.pkl"
            with open(instance_file_name, 'rb') as f:
                instance_dataset = pickle.load(f)

            eva = TSPGLS(instance_dataset, size, n_test_ins, n_jobs=22)

            iden_score, iden_results = eva.eval(ide)

            time_start = time.time()
            score, results = eva.eval(alg)

            time_res = time.time() - time_start
            # ortool_res = np.average(results['ortool'])
            ortool_res = np.average([ort['max_distance'] for ort in results['ortool']])
            identity_res = np.average(iden_results['heuristic'])
            i_score_res = np.average(iden_score)
            heuristic_res = np.average(results['heuristic'])
            h_score_res = score

            result = (
                f"Instance size {size}:\n"
                f" - Time taken: {time_res:.3f}\n"
                f" - OR-Tool avg time: {ortool_res:.3f}\n"
                f" - Identity avg time: {identity_res:.3f}\n"
                f" - Identity Score: {i_score_res:.3f}\n"
                f" - Heuristic avg time: {heuristic_res:.3f}\n"
                f" - Heuristic Score: {h_score_res:.3f}\n"
            )
            print(result)
            file.write(result + "\n")

            times.append(time_res)
            ortool.append(ortool_res)
            identi.append(identity_res)
            i_score.append(i_score_res)
            heuristic.append(heuristic_res)
            h_score.append(h_score_res)

            # NOTE: due to multithreading the order of ortool might be different (so take great care when comparing heuristic to ortool)
            plot_i_score.append([g / o - 1.0 for g, o in zip(iden_results['heuristic'], iden_results['ortool'])])
            plot_h_score.append([h / o - 1.0 for h, o in zip(results['heuristic'], results['ortool'])])

        print("Time (Sec)" + "".join(f" & {x:.3f}" for x in times) + " \\\\")
        print("OR-Tool" + "".join(f" & {x:.3f}" for x in ortool) + " \\\\")
        print("Greedy" + "".join(f" & {x:.3f}" for x in identi) + " \\\\")
        print("Score" + "".join(f" & {x:.3f}" for x in i_score) + " \\\\")
        print("Heuristic" + "".join(f" & {x:.3f}" for x in heuristic) + " \\\\")
        print("Score" + "".join(f" & {x:.3f}" for x in h_score) + " \\\\")

        sizes = [10, 20, 50, 100, 200]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.set_title("Identity")
        ax1.boxplot(plot_i_score, tick_labels=sizes)

        ax2.set_title("Heuristic")
        ax2.boxplot(plot_h_score, tick_labels=sizes)

        ax2.sharey(ax1)
        plt.show()
