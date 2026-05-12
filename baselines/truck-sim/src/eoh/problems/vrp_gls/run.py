import copy
import numpy as np
from itertools import pairwise

from src.eoh.problems.interface import PromptsBase, ProblemBase
from src.eoh.problems.vrp_gls.gls import GuidedLocalSearch


class Prompts(PromptsBase):
    def __init__(self):
        super().__init__()
        # self._prompt_task: str = (
        #     "Given a set of nodes with their coordinates and 3 trucks, "
        #     "you need to find the routes for each truck such altogether they visit every node once and returns to the starting node. "
        #     "The task can be solved step-by-step by starting from the current node and iteratively choosing the next node every time a truck arrives at a node. "
        #     "Help me design a novel algorithm that is different from the algorithms in literature to select this next node in each step."
        # )

        # self._prompt_task: str = (
        #     "With a fleet of 3 trucks, you are tasked with finding routes to visit a list of nodes and return to the start. "
        #     "This task can be solved step-by-step by iteratively choosing the next node every time a truck arrives at a node. "
        #     "There is also the option for the truck to wait at a node for a period of time if 'None' is returned. "
        #     "Help me design a novel algorithm that is different from the algorithms in literature."
        # )
        self._prompt_task = (
            "Task: Given an edge distance matrix and a local optimal routes, "
            "please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum "
            "with the final goal of minimizing the length of the longest route. "
            "You should create a heuristic for me to update the edge distance matrix."
        )
        self._prompt_task_ext = (
            "Task: Given an edge distance matrix and a local optimal routes, "
            "please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum "
            "with the final goal of minimizing the length of the longest route. "
            "You should create a heuristic for me to update the edge distance matrix."
        )
        self._prompt_func_name: str = "update_edge_distance"
        self._prompt_func_inputs = ['edge_distance',
                                    'local_opt_tours',
                                    'edge_n_used']
        self._prompt_func_outputs = ['updated_edge_distance']
        self._prompt_inout_inf = (
            "'local_opt_tours' includes the local optimal tours of IDs, "
            "'edge_distance' and 'edge_n_used' are matrices, "
            "'edge_n_used' includes the number of each edge used during permutation."
        )
        self._prompt_other_inf = "All are Numpy arrays."


TRUCK_NUM = 3


class VRPGLS(ProblemBase):
    def __init__(self, data, size: int, n_test: int):
        super().__init__()
        self.ndelay = 1
        self.neighbor_size = np.minimum(50, size)  # TODO: do we want to only pass in only the 50 best neighbors?
        self.running_time = 10

        self.instance_data = data
        # self.problem_size = 10
        # self.n_instance = 1
        self.problem_size = size
        self.n_instance = n_test

        self.prompts: PromptsBase = Prompts()

    def eval(self, eva):
        # dis = np.zeros((TRUCK_NUM, self.n_instance))
        results = {
            "heuristic": [],
            "ortool": [],
            "routes": [],
        }
        scores = []
        n_ins = 0

        for instance, distance_mtx, ortool in self.instance_data:
            if n_ins == self.n_instance:
                break

            gls = GuidedLocalSearch(eva,
                                    self.problem_size,
                                    instance,
                                    distance_mtx,
                                    max_iterations=1000,
                                    running_time=self.running_time)
                                    # , alpha=1.0, penalty_multiplier=10)
            best_solution = gls.search()
            best_solution.calculate_cost(distance_mtx)

            route_lens = []
            for i, route in enumerate(best_solution.routes):
                distance: float = 0
                for c, n in pairwise(route):
                    distance += distance_mtx[c, n]
                route_lens.append(distance)

            scores.append(max(route_lens) / ortool["max_distance"])
            results["heuristic"].append(max(route_lens))
            results["ortool"].append(ortool)
            results["routes"].append([r for r in best_solution.routes])

        score = (np.average(scores) - 1) * 100
        return score, results
