import json
import logging
import argparse
import os
import sys
import numpy as np

from statistics import mean
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

from src.policy.fcfs import FirstComeFirstServed
from src.policy.greedy import CheapestInsertion
from src.policy.ortool import OrTool

from src.simulator.controller import Controller


# TODO LATER: implement trailer drop off (low prio)


def setup_logging(verbose: bool):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)


def collect_dataset_paths(path: str) -> list[str]:
    dataset_paths = []

    if os.path.isdir(path):
        for file in os.listdir(path):
            filename = os.fsdecode(file)

            new_path = os.path.join(path, filename)
            if os.path.isfile(new_path) and filename.endswith(".json"):
                dataset_paths.append(new_path)

    else:
        if path.endswith(".json"):
            dataset_paths.append(path)

    return dataset_paths


policy_dict = {
    'fcfs': FirstComeFirstServed,
    'greedy': CheapestInsertion,
    'ortool': OrTool,
}

SIM_TIMESTEP = 5.0

def run_experiment(path: str, policy: str, animated: bool):
    with open(path, 'r') as f:
        data = json.load(f)

    Controller.full_reset()
    con = Controller(
        data,
        policy_dict[policy],
        animated)

    update_func = lambda x: con.update(float(x) * SIM_TIMESTEP)
    time_steps: int = int(data["end of day"] / SIM_TIMESTEP)

    if animated:
        fig = con.get_fig()
        ani = animation.FuncAnimation(
            fig=fig,
            func=update_func,
            frames=time_steps,
            interval=100)

        base = os.path.basename(path)
        head, tail = os.path.splitext(base)

        assert tail == ".json"
        assert os.path.exists("output/")

        ani.save(f"output/{head}.mp4", dpi=300, writer=FFMpegWriter(fps=2))

    else:
        for t in range(time_steps):
            update_func(t)

    # get results then sort them by order of priority (sort is stable, i.e. will preserve order of equal elements)
    exp_results = [truck.get_stats() for truck in con.trucks]
    exp_results.sort(key=lambda x: x["distance"])
    exp_results.sort(key=lambda x: x["time"])
    exp_results.sort(key=lambda x: x["count"], reverse=True)
    return exp_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path of dataset to load', type=str)
    parser.add_argument('policy',
                        choices=policy_dict.keys(), type=str)
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action="store_true")
    parser.add_argument('-a', '--animation',
                        help='Start and save animation', action='store_true')
    args = parser.parse_args()

    setup_logging(args.verbose)

    paths = collect_dataset_paths(args.path)
    logging.debug(f"Collected dataset paths: {paths}")

    if len(paths) == 0:
        logging.info("Nothing to do, exiting")
        sys.exit(0)

    if args.animation and len(paths) > 1:
        logging.info("Animation takes a long time")
        sys.exit(1)

    all_results = [run_experiment(p, args.policy, args.animation) for p in paths]

    count_results: list[list[int]] = [[res["count"] for res in result] for result in all_results]
    time_results: list[list[float]] = [[res["time"] for res in result] for result in all_results]
    distance_results: list[list[float]] = [[res["distance"] for res in result] for result in all_results]

    res_str = f"Result from {len(paths)} experiments with {args.policy}:\n"

    tot_times = [sum(exp) for exp in count_results]
    avg_tot_deli = mean(tot_times)
    std_tot_deli = np.std(tot_times)
    res_str += f" - Average deliveries: {avg_tot_deli:.3f} (std: {std_tot_deli:.2f})\n"

    end_times = [max(exp) for exp in time_results]
    avg_end_time = mean(end_times)
    std_end_time = np.std(end_times)
    res_str += f" - Average end time: {avg_end_time:.3f} min (std: {std_end_time:.2f})\n"

    tot_times = [sum(exp) for exp in time_results]
    avg_tot_time = mean(tot_times)
    std_tot_time = np.std(tot_times)
    res_str += f" - Average total time: {avg_tot_time:.3f} min (std: {std_tot_time:.2f})\n"

    tot_dists = [sum(exp) / 1000 for exp in distance_results]
    avg_tot_dist = mean(tot_dists)
    std_tot_dist = np.std(tot_dists)
    res_str += f" - Average total distance: {avg_tot_dist:.3f} km (std: {std_tot_dist:.2f})\n"

    print(res_str)
