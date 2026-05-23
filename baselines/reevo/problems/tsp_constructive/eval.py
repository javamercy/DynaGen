import math
from os import path
import numpy as np
import sys
from scipy.spatial import distance_matrix
import logging
from copy import copy
import os

try:
    from gpt import select_next_node_v2 as select_next_node
except:
    from gpt import select_next_node


def eval_heuristic(node_positions: np.ndarray) -> float:
    '''
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.
    
    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).
    
    Returns
    -------
    obj : float
        The length of the generated tour.
    '''
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    start_node = 0
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in range(problem_size - 1):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj


def eval_tsplib_instance(file_path: str) -> tuple[str, float | None, float]:
    from dynagen.domain.tsp_parser import load_tsplib_file

    inst = load_tsplib_file(file_path)
    tour_length = eval_heuristic(inst.coordinates)
    return file_path, inst.optimal_length, tour_length
    

if __name__ == '__main__':
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    basepath = path.join(path.dirname(__file__), "dataset")
    if not path.isfile(path.join(basepath, "train500_dataset.npy")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        import multiprocessing as mp
        from functools import partial
        dataset_path = path.join(basepath, f"train{problem_size}_dataset.npy")
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        with mp.Pool() as pool:
            objs = pool.map(eval_heuristic, node_positions)
            
        for i, obj in enumerate(objs):
            print(f"[*] Instance {i}: {obj}")
        
        print("[*] Average:")
        print(np.mean(objs))
    
    else:
        import glob
        import multiprocessing as mp
        sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), "../../../../")))
        
        test_dir = path.abspath(path.join(path.dirname(__file__), "../../../../data/tsp/test_instances"))
        test_files = glob.glob(path.join(test_dir, "*.tsp"))
        logging.info(f"[*] Evaluating on {len(test_files)} TSPLIB instances from {test_dir}")
        
        gaps = []
        with mp.Pool() as pool:
            results = pool.map(eval_tsplib_instance, sorted(test_files))

        for file_path, optimal_length, tour_length in results:
            if optimal_length is not None and optimal_length > 0:
                gap = (tour_length - optimal_length) / optimal_length * 100
                gaps.append(gap)
                print(f"[*] Instance {path.basename(file_path)}: Tour Length = {tour_length:.2f}, Optimal = {optimal_length}, Gap = {gap:.2f}%")
            else:
                print(f"[*] Instance {path.basename(file_path)}: Tour Length = {tour_length:.2f}, Optimal = {optimal_length} (No gap calculated)")

        if gaps:
            print(f"[*] Average Gap: {np.mean(gaps):.2f}%")
