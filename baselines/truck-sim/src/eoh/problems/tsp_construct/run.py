import numpy as np

from src.eoh.problems.interface import PromptsBase, ProblemBase


class Prompts(PromptsBase):
    def __init__(self):
        super().__init__()

        self._prompt_task: str = (
            "Given a set of nodes with their coordinates, you need to find the shortest route that visits each node once and returns to the starting node. "
            "The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. "
            "Help me design a novel algorithm that is different from the algorithms in literature to select the next node in each step."
        )
        self._prompt_task_ext: str = (
            "Given a set of nodes with their coordinates, you need to find the shortest route that visits each node once and returns to the starting node. "
            "The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. "
            "Help me design an algorithm to select the next node in each step."
        )

        self._prompt_func_name = "select_next_node"
        self._prompt_func_inputs = ["current_node", "destination_node", "unvisited_nodes", "distance_matrix"]
        self._prompt_func_outputs = ["next_node"]
        self._prompt_inout_inf = (
            "'current_node', 'destination_node', 'next_node', and 'unvisited_nodes' are node IDs. "
            "'distance_matrix' is the distance matrix of nodes."
        )
        self._prompt_other_inf = "All are Numpy arrays."


class TSPConstruct(ProblemBase):
    def __init__(self, data, size: int, n_test: int):
        super().__init__()

        # ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        # sys.path.append(ABS_PATH)  # This is for finding all the modules
        # Construct the absolute path to the pickle file
        #pickle_file_path = os.path.join(ABS_PATH, 'instances.pkl')

        # with open("./instances.pkl" , 'rb') as f:
        #     self.instance_data = pickle.load(f)
        self.ndelay = 1
        self.neighbor_size = np.minimum(50, size)
        self.running_time = 10

        self.problem_size = size
        self.instance_data = data
        self.n_instance = n_test

        self.prompts: PromptsBase = Prompts()

    #@func_set_timeout(5)
    def eval(self, eva):
        dis = np.ones(self.n_instance)
        results = {
            "heuristic": [],
            "ortool": [],
            "routes": [],
        }
        scores = []
        n_ins = 0

        for instance, distance_matrix, ortool in self.instance_data:
            neighbor_matrix = self.generate_neighborhood_matrix(instance)
            destination_node = 0
            current_node = 0

            route = np.zeros(self.problem_size)

            for i in range(1,self.problem_size-1):
                near_nodes = neighbor_matrix[current_node][1:]
                mask = ~np.isin(near_nodes, route[:i])

                unvisited_near_nodes = near_nodes[mask]
                unvisited_near_size = np.minimum(self.neighbor_size,unvisited_near_nodes.size)
                unvisited_near_nodes = unvisited_near_nodes[:unvisited_near_size]

                next_node = eva.select_next_node(current_node,
                                                 destination_node,
                                                 unvisited_near_nodes,
                                                 distance_matrix)

                if next_node in route:
                    # print("wrong algorithm select duplicate node, retrying ...")
                    return None

                current_node = next_node
                route[i] = current_node

            mask = ~np.isin(np.arange(self.problem_size),route[:self.problem_size-1])

            last_node = np.arange(self.problem_size)[mask]
            current_node = last_node[0]
            route[self.problem_size-1] = current_node

            LLM_dis = self.tour_cost(instance, route)
            dis[n_ins] = LLM_dis

            scores.append(LLM_dis / ortool["max_distance"])

            results["heuristic"].append(LLM_dis)
            results["ortool"].append(ortool)
            results["routes"].append([route])

            n_ins += 1
            if n_ins == self.n_instance:
                break
            #self.route_plot(instance,route,self.oracle[n_ins])

        ave_dis = np.average(dis)

        score = (np.average(scores) - 1) * 100
        return score, results


if __name__ == "__main__":
    prompts = Prompts()
    print(prompts)
