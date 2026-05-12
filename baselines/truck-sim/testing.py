import os
import sys
import json
import time
import types
import pickle
import random
import resource
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

from src.eoh.utils.getParas import Paras
from src.eoh.evolution import methods
from src.eoh.problems import problems


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# class EVOL:
#     def __init__(self, paras, prob=None):
#         print("-----------------------------------------\n"
#               "---              Start EoH            ---\n"
#               "-----------------------------------------")
#
#         createFolders.create_folders(paras.exp_output_path)
#         print("- output folder created -")
#
#         self.paras = paras
#         print("-  parameters loaded -")
#
#         self.prob = prob
#         random.seed(2024)  # set a random seed
#
#     def run(self, instances, size: int, n_test: int):
#         problem_generator = problems.Probs(self.paras.problem)
#         problem = problem_generator.get_problem(instances, size, n_test)
#
#         method_generator = methods.Methods(self.paras, problem)
#         method = method_generator.get_method()
#
#         method.run()
#
#         print("> End of Evolution! ")
#         print("-----------------------------------------\n"
#               "---     EoH successfully finished !   ---\n"
#               "-----------------------------------------")

# 0. generate testing data

# 1. run EoH and save results to results
# 2. extract the best performing algorithm
# 3. create a folder called results/eval
# 4. run eval and save ALL results to eval
# 5. move result based on gpt version used

def evaluate(eva, code_string):
    # set memory limit to avoid getting killed by OS
    soft_limit_bytes = 4 * 1024 * 1024 * 1024  # 4 GB
    hard_limit_bytes = 5 * 1024 * 1024 * 1024  # 5 GB
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit_bytes, hard_limit_bytes))

    # noinspection PyBroadException
    try:
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Create a new module object
            heuristic_module = types.ModuleType("heuristic_module")

            # Execute the code string in the new module's namespace
            exec(code_string, heuristic_module.__dict__)

            # Add the module to sys.modules so it can be imported
            sys.modules[heuristic_module.__name__] = heuristic_module

            # Now you can use the module as you would any other
            # score, _ = func_timeout(10, lambda: self.eval(heuristic_module))
            return eva.eval(heuristic_module)

    except Exception as e:
        # print(code_string)
        # print(traceback.format_exc())

        # print("Error:", str(e))
        return None, {}


api_key = ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Handler interface for running experiments.')
    parser.add_argument('exp',
                        type=str, help='problem to run experiments for')
    parser.add_argument('dataset',
                        type=str, help='which dataset to use for testing and training')
    parser.add_argument('-m', '--model',
                        type=str, default='gpt-3.5-turbo', help='model to use to generate heuristic')
    parser.add_argument('--mini',
                        action='store_true', default=False, help='perform a miniature test run')
    parser.add_argument('-n', '--n_threads',
                        type=int, default=1, help='number of threads')
    parser.add_argument('--eval',
                        action='store_true', default=False, help='just evaluate the results')
    args = parser.parse_args()

    short_name = args.exp.split('_')[0]
    short_type = args.exp.split('_')[1]
    long_name = args.exp

    print("-----------------------------------------\n"
          "---              Start EoH            ---\n"
          "-----------------------------------------")

    # Parameter initialization
    paras = Paras()

    # Set parameters
    paras.set_paras(method="eoh",
                    problem=args.exp,
                    llm_api_endpoint="api.openai.com",
                    llm_api_key=api_key,
                    llm_model=args.model,
                    ec_pop_size=10,  # number of samples in each population
                    ec_n_pop=20,  # number of populations
                    # ec_pop_size=5,  # number of samples in each population
                    # ec_n_pop=10,  # number of populations
                    # ec_pop_size=5,  # number of samples in each population
                    # ec_n_pop=10,  # number of populations
                    exp_n_proc=8,  # multi-core parallel
                    exp_extended_init=True,
                    exp_strict_init=True,
                    exp_debug_mode=False,
                    eva_numba_decorator=False,
                    eva_timeout=180,  # Set the maximum evaluation time for each heuristic (increase it if more instances are used for evaluation)
                    )

    if args.mini:
        paras.ec_pop_size = 2
        paras.ec_n_pop = 2

    print("-  parameters loaded -")

    # Check if the results folder already exists
    results = os.path.join("./testing", short_name, long_name, "results_" + args.model)
    if not os.path.exists(results):
        os.makedirs(results)
    # else:
        # print(f"Error: already exists {results}")
        # exit(1)

    # Create subfolders
    subfolders = ["pops", "pops_best", "eval"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(results, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    print("- output folder created -")

    paras.exp_output_path = results

    # Load training dataset
    n_test_ins = 8
    instance_file_name = f"./testing/{short_name}/training_data_{args.dataset}/instances.pkl"
    print(f"Loading training dataset from {instance_file_name}")
    if not os.path.exists(instance_file_name):
        print(f"Error: path does not exist {instance_file_name}")
        exit(1)
    with open(instance_file_name, 'rb') as f:
        instance_dataset = pickle.load(f)
    print("- training dataset loaded -")

    # Load the problem
    random.seed(2024)  # set a random seed
    problem_generator = problems.Probs(paras.problem)
    problem = problem_generator.get_problem(instance_dataset, 50, n_test_ins)

    train_start = time.time()
    if not args.eval:
        # Run EoH
        method_generator = methods.Methods(paras, problem)
        method = method_generator.get_method()
        method.run()
    train_time = time.time() - train_start

    print("> End of Evolution! ")
    print("-----------------------------------------\n"
          "---     EoH successfully finished !   ---\n"
          "-----------------------------------------")

    files = [file.name for file in os.scandir(os.path.join(results, "pops_best")) if file.is_file()]
    if "population_generation_1.json" not in files:
        print("Error: not results produced?")
        exit(1)
    last_generation = "population_generation_1.json"
    for i in range(50):  # THIS IS AWFUL
        if f"population_generation_{i}.json" in files:
            last_generation = f"population_generation_{i}.json"


    # last_generation = "population_generation_10.json"


    pop_file = os.path.join(results, "pops_best", last_generation)
    print(f"Using population from {pop_file}")
    with open(pop_file, 'r') as f:
        heuristic_code_string = json.load(f)["code"]

    if short_type == "construct":
        baseline_path = os.path.join("./testing", short_name, long_name, "greedy.py")
    elif short_type == "gls":
        baseline_path = os.path.join("./testing", short_name, long_name, "identity.py")
    else:
        print("Error: unknown type for problem solution")
        exit(1)
    with open(baseline_path, 'r') as file:
        baseline_code_string = file.read()

    problem_size = [10, 20, 50, 100, 200]
    n_test_ins = 64
    print("Start evaluation...")

    o_avg_max_dis = []
    o_missed_p = []  # backwards compatibility
    o_feasible = []

    b_eval_times = []
    b_avg_max_dis = []
    b_avg_scores = []
    b_all_scores = []
    b_missed_p = []
    b_feasible = []

    h_eval_times = []
    h_avg_max_dis = []
    h_avg_scores = []
    h_all_scores = []
    h_missed_p = []
    h_feasible = []

    eval_results = os.path.join(results, "eval")
    with (open(os.path.join(eval_results, "results.txt"), "w") as file):
        file.write(f"Train time: {train_time}\n")

        baseline_results = {}
        heuristic_results = {}

        for size in problem_size:
            instance_file_name = f"./testing/{short_name}/testing_data_{args.dataset}/instance_data_{size}.pkl"
            with open(instance_file_name, 'rb') as f:
                instance_dataset = pickle.load(f)

            eva = problem_generator.get_problem(instance_dataset, size, n_test_ins)

            b_time_start = time.time()
            # b_score, b_results = evaluate(eva, baseline_code_string)
            _, b_all_results = evaluate(eva, baseline_code_string)
            b_time_res = time.time() - b_time_start

            h_time_start = time.time()
            # h_score, h_results = evaluate(eva, heuristic_code_string)
            _, h_all_results = evaluate(eva, heuristic_code_string)
            h_time_res = time.time() - h_time_start

            def unpack_results(results):
                missed = 0
                unfeasible = 0
                scores = []
                max_diss = []

                for i in range(n_test_ins):
                    ortool = results["ortool"][i]
                    # assert ortool['missed'] == 0
                    ort_max_dis = ortool["max_distance"]

                    max_dis = results["heuristic"][i]
                    if "missed" in results.keys():
                        missed += results["missed"][i]
                        if results["missed"][i] > 0:
                            unfeasible += 1

                    max_diss.append(max_dis)
                    scores.append((max_dis / ort_max_dis - 1) * 100)

                if len(max_diss) == 0:
                    max_diss.append(0)
                if len(scores) == 0:
                    scores.append(0)

                return missed, unfeasible, max_diss, scores

            # if "missed" in h_all_results["ortool"][0].keys():
            #     o_results = [ort["max_distance"] for ort in h_all_results["ortool"] if ort["missed"] == 0]
            # else:
            #     o_results = [ort["max_distance"] for ort in h_all_results["ortool"]]
            o_result = np.average([ort["max_distance"] for ort in h_all_results["ortool"]])
            o_missed = 0
            o_unfeasible = 0
            if "missed" in h_all_results["ortool"][0].keys():
                for i in range(n_test_ins):
                    o_missed += h_all_results["ortool"][i]["missed"]
                    if h_all_results["ortool"][i]["missed"] > 0:
                        o_unfeasible += 1
            o_missed_percent = (o_missed / (n_test_ins * size)) * 100
            o_success_percent = (1 - o_unfeasible/n_test_ins) * 100

            b_missed, b_unfeasible, b_results, b_scores = unpack_results(b_all_results)
            b_result = np.average(b_results)
            b_score = np.average(b_scores)
            b_missed_percent = (b_missed / (n_test_ins * size)) * 100
            b_success_percent = (1 - b_unfeasible/n_test_ins) * 100

            h_missed, h_unfeasible, h_results, h_scores = unpack_results(h_all_results)
            h_result = np.average(h_results)
            h_score = np.average(h_scores)
            h_missed_percent = (h_missed / (n_test_ins * size)) * 100
            h_success_percent = (1 - h_unfeasible/n_test_ins) * 100

            try:
                header = f"Instance size {size} with {len(h_all_results['ortool'][0]['routes'])} trucks:\n"
            except:
                header = f"Instance size {size}:\n"
            result = (
                header +
                f" - OR-Tool avg dist: {o_result:.3f}\n"
                f" - OR-Tool missed (%): {o_missed_percent:.3f}\n"
                f" - OR-Tool feasible (%): {o_success_percent:.3f}\n"
                f" - Baseline avg dist: {b_result:.3f}\n"
                f" - Baseline avg score: {b_score:.3f}\n"
                f" - Baseline missed (%): {b_missed_percent:.3f}\n"
                f" - Baseline feasible (%): {b_success_percent:.3f}\n"
                f" - Time taken: {b_time_res:.3f}\n"
                f" - Heuristic avg dist: {h_result:.3f}\n"
                f" - Heuristic avg score: {h_score:.3f}\n"
                f" - Heuristic missed (%): {h_missed_percent:.3f}\n"
                f" - Heuristic feasible (%): {h_success_percent:.3f}\n"
                f" - Time taken: {h_time_res:.3f}\n"
            )
            print(result)
            file.write(result + "\n")

            o_avg_max_dis.append(o_result)
            o_missed_p.append(o_missed_percent)
            o_feasible.append(o_success_percent)

            b_eval_times.append(b_time_res)
            b_avg_max_dis.append(b_result)
            b_avg_scores.append(b_score)
            b_all_scores.append(b_scores)
            b_missed_p.append(b_missed_percent)
            b_feasible.append(b_success_percent)

            h_eval_times.append(h_time_res)
            h_avg_max_dis.append(h_result)
            h_avg_scores.append(h_score)
            h_all_scores.append(h_scores)
            h_missed_p.append(h_missed_percent)
            h_feasible.append(h_success_percent)

    with open(os.path.join(eval_results, "baseline_results.json"), "w") as file:
        file.write(json.dumps(baseline_results, indent=4, cls=NumpyEncoder))
    with open(os.path.join(eval_results, "heuristic_results.json"), "w") as file:
        file.write(json.dumps(heuristic_results, indent=4, cls=NumpyEncoder))

    # save this stuff
    latex = (
        "OR-Tool"             + "".join(f" & {x:.3f}" for x in o_avg_max_dis) + " \\\\\n"
        "OR-Tool missed"      + "".join(f" & {x:.3f}" for x in o_missed_p) + " \\\\\n"
        "OR-Tool feasible"    + "".join(f" & {x:.3f}" for x in o_feasible) + " \\\\\n"
        "Baseline"            + "".join(f" & {x:.3f}" for x in b_avg_max_dis) + " \\\\\n"
        "Baseline score"      + "".join(f" & {x:.3f}" for x in b_avg_scores) + " \\\\\n"
        "Baseline missed"     + "".join(f" & {x:.3f}" for x in b_missed_p) + " \\\\\n"
        "Baseline feasible"   + "".join(f" & {x:.3f}" for x in b_feasible) + " \\\\\n"
        "Time (Sec)"          + "".join(f" & {x:.3f}" for x in b_eval_times) + " \\\\\n"
        "Heuristic"           + "".join(f" & {x:.3f}" for x in h_avg_max_dis) + " \\\\\n"
        "Heuristic score"     + "".join(f" & {x:.3f}" for x in h_avg_scores) + " \\\\\n"
        "Heuristic missed"    + "".join(f" & {x:.3f}" for x in h_missed_p) + " \\\\\n"
        "Heuristic feasible"  + "".join(f" & {x:.3f}" for x in h_feasible) + " \\\\\n"
        "Time (Sec)"          + "".join(f" & {x:.3f}" for x in h_eval_times) + " \\\\\n"
    )
    with open(os.path.join(eval_results, "latex.txt"), "w") as file:
        file.write(latex)

    # https://stackoverflow.com/a/63243881
    sizes = [10, 20, 50, 100, 200]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.set_title("Baseline")
    ax1.boxplot(b_all_scores, tick_labels=sizes)

    ax2.set_title("Heuristic")
    ax2.boxplot(h_all_scores, tick_labels=sizes)

    # save to path
    ax2.sharey(ax1)
    plt.savefig(os.path.join(eval_results, "results.png"))
