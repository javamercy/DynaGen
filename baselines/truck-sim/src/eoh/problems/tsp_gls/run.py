import time
import multiprocessing
import numpy as np
from joblib import Parallel, delayed

from .gls import gls_evol, utils
from src.eoh.problems.interface import PromptsBase, ProblemBase


class Prompts(PromptsBase):
    def __init__(self):
        super().__init__()

        self._prompt_task = (
            "Task: Given an edge distance matrix and a local optimal route, "
            "please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum "
            "with the final goal of finding a tour with minimized distance. "
            "You should create a heuristic for me to update the edge distance matrix."
        )
        self._prompt_task_ext = (
            "Task: Given an edge distance matrix and a local optimal route, "
            "please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum "
            "with the final goal of finding a tour with minimized distance. "
            "You should create a heuristic for me to update the edge distance matrix."
        )
        self._prompt_func_name = "update_edge_distance"
        self._prompt_func_inputs = ['edge_distance',
                                    'local_opt_tour',
                                    'edge_n_used']
        self._prompt_func_outputs = ['updated_edge_distance']
        self._prompt_inout_inf = (
            "'local_opt_tour' includes the local optimal tour of IDs, "
            "'edge_distance' and 'edge_n_used' are matrices, "
            "'edge_n_used' includes the number of each edge used during permutation."
        )
        self._prompt_other_inf = "All are Numpy arrays."

#     def get_prompt_create(self):
#         prompt_content = "Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
# You should create a strategy for me to update the edge distance matrix. \
# Provide a description of the new strategy in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new strategy. The code is a Python function called 'update_edge_distance' that takes three inputs 'edge_distance', 'local_opt_tour', 'edge_n_used', and outputs the 'updated_edge_distance', \
# where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation. All are Numpy arrays. Pay attention to the format and do not give additional explanation."
#         return prompt_content


#     def get_prompt_crossover(self,indiv1,indiv2):
#         prompt_content = "Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
# I have two strategies with their codes to update the distance matrix. \
# The first strategy and the corresponding code are: \n\
# Strategy description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# The second strategy and the corresponding code are: \n\
# Strategy description: "+indiv2['algorithm']+"\n\
# Code:\n\
# "+indiv2['code']+"\n\
# Please help me create a new strategy that is totally different from them but can be motivated from them. \
# Provide a description of the new strategy in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new strategy. The code is a Python function called 'update_edge_distance' that takes three inputs 'edge_distance', 'local_opt_tour', 'edge_n_used', and outputs the 'updated_edge_distance', \
# where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation. All are Numpy arrays. Pay attention to the format and do not give additional explanation."
#         return prompt_content

#     def get_prompt_mutation(self,indiv1):
#         prompt_content = "Task: Given a set of nodes with their coordinates, \
# you need to find the shortest route that visits each node once and returns to the starting node. \
# The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. \
# I have a strategy with its code to select the next node in each step as follows. \
# Strategy description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# Please assist me in creating a modified version of the strategy provided. \
# Provide a description of the new strategy in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new strategy. The code is a Python function called 'update_edge_distance' that takes three inputs 'edge_distance', 'local_opt_tour', 'edge_n_used', and outputs the 'updated_edge_distance', \
# where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation. All are Numpy arrays. Pay attention to the format and do not give additional explanation."
#         return prompt_content


class TSPGLS(ProblemBase):
    def __init__(self, data, size: int, n_test: int, n_jobs: int = 1):
        super().__init__()
        # self.n_inst_eva = 3 # a small value for test only
        # self.time_limit = 10 # maximum 10 seconds for each instance
        # self.ite_max = 1000 # maximum number of local searches in GLS for each instance
        # self.perturbation_moves = 1 # movers of each edge in each perturbation
        # path = os.path.dirname(os.path.abspath(__file__))
        # self.instance_path = path+'/TrainingData/TSPAEL64.pkl' #,instances=None,instances_name=None,instances_scale=None
        # self.debug_mode=False

        # self.coords, self.instances, self.opt_costs = readTSPRandom.read_instance_all(self.instance_path)

        self.ndelay = 1
        self.n_jobs = n_jobs

        self.running_time = 10  # maximum 10 seconds for each instance
        self.ite_max = 1000  # maximum number of local searches in GLS for each instance
        self.perturbation_moves = 1  # movers of each edge in each perturbation

        self.problem_size = size
        self.instance_data = data
        self.n_instance = n_test

        self.prompts: PromptsBase = Prompts()

    def eval(self, eva):
        results = {
            "heuristic": [],
            "ortool": [],
            "routes": [],
        }

        def perform_eval(instance, distance_matrix, ortool):
            try:
                time.sleep(1)
                t = time.time()

                # get initial solution
                init_tour = gls_evol.nearest_neighbor_2End(distance_matrix, 0).astype(int)
                init_cost = utils.tour_cost_2End(distance_matrix, init_tour)
                nb = 100
                nearest_indices = np.argsort(distance_matrix, axis=1)[:, 1:nb + 1].astype(int)

                best_tour, best_cost, iter_i = gls_evol.guided_local_search(instance,
                                                                            distance_matrix,
                                                                            nearest_indices,
                                                                            init_tour,
                                                                            init_cost,
                                                                            t + self.running_time,
                                                                            self.ite_max,
                                                                            self.perturbation_moves,
                                                                            first_improvement=False,
                                                                            guide_algorithm=eva)

                # # weirdness with locks
                # results["heuristic"].append(best_cost)
                # results["ortool"].append(ortool["max_distance"])
                # results["routes"].append([best_tour])

                return best_cost / ortool["max_distance"], (best_cost, ortool, [best_tour])

            except Exception as e:
                # print("Error:", str(e))  # Print the error message
                return 1E10, (1E10, ortool["max_distance"], [])

        # scores = []
        # for instance, distance_matrix, ortool in self.instance_data:
        #     scores.append(perform_eval(instance, distance_matrix, ortool))

        scores_and_results = Parallel(n_jobs=self.n_jobs, timeout=self.running_time + 5)(
            delayed(perform_eval)(instance, distance_matrix, ortool)
            for instance, distance_matrix, ortool in self.instance_data)

        scores = []
        for score, result in scores_and_results:
            scores.append(score)
            results["heuristic"].append(result[0])
            results["ortool"].append(result[1])
            results["routes"].append(result[2])

        score = (np.average(scores) - 1) * 100
        return score, results
