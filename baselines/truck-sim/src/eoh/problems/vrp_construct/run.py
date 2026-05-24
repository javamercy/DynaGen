import numpy as np

from src.eoh.problems.interface import PromptsBase, ProblemBase, truck_num_scaling


class Prompts(PromptsBase):
    def __init__(self):
        super().__init__()

        # self._prompt_task: str = (
        #     "Given a set of nodes with their coordinates and 3 trucks, "
        #     "you need to find the routes for each truck such altogether they visit every node once and returns to the starting node. "
        #     "The task can be solved step-by-step by starting from the current node and iteratively choosing the next node every time a truck arrives at a node. "
        #     "Help me design a novel algorithm that is different from the algorithms in literature to select this next node in each step."
        # )
        self._prompt_task: str = (
            "With a fleet of trucks, you are tasked with finding routes to visit a list of nodes and return to the start. "
            "This task can be solved step-by-step by iteratively choosing the next node every time a truck arrives at a node. "
            "There is also the option for the truck to wait at a node for a period of time if 'None' is returned. "
            "Help me design a novel algorithm that is different from the algorithms in literature."
        )
        self._prompt_task_ext: str = (
            "With a fleet of trucks, you are tasked with finding routes to visit a list of nodes and return to the start. "
            "This task can be solved step-by-step by iteratively choosing the next node every time a truck arrives at a node. "
            "There is also the option for the truck to wait at a node for a period of time if 'None' is returned. "
        )

        self._prompt_func_name: str = "select_next_node"
        self._prompt_func_inputs: list[str] = ["current_node",
                                               "truck_nodes"
                                               "depot_node",
                                               "unvisited_nodes",
                                               "distance_matrix"]
        self._prompt_func_outputs: list[str] = ["next_node"]
        self._prompt_inout_inf: str = (
            "'current_node', 'depot_node', 'next_node', and 'unvisited_nodes' are node IDs. "
            "'truck_nodes' are the next nodes that each truck is moving towards (or is already there). "
            "'distance_matrix' is the distance matrix of nodes."
        )
        self._prompt_other_inf: str = "All are Numpy arrays."


class VRPConstruct(ProblemBase):
    def __init__(self, data, size: int, n_test: int):
        super().__init__()
        self.neighbor_size = np.minimum(50, size)
        self.running_time = 10

        self.problem_size = size
        self.instance_data = data
        self.n_instance = n_test
        self.truck_num = truck_num_scaling(size)

        self.prompts: PromptsBase = Prompts()

    def eval(self, eva):
        dis = np.zeros((self.truck_num, self.n_instance))
        results = {
            "heuristic": [],
            "ortool": [],
            "routes": [],
        }
        scores = []
        n_ins = 0

        for instance, distance_matrix, ortool in self.instance_data:
            if n_ins == self.n_instance:
                break

            # get neighborhood matrix
            neighbor_matrix = self.generate_neighborhood_matrix(instance)
            destination_node = 0
            distance_left = np.zeros(self.truck_num)
            current_node = np.zeros(self.truck_num, dtype=int)
            routes = [[0] for _ in range(self.truck_num)]

            for i in range(1, self.problem_size):
                cur_truck = np.argmin(distance_left)
                # print(cur_truck)
                distance_left -= distance_left[cur_truck]
                distance_left[cur_truck] = 0

                cur_node = current_node[cur_truck]
                near_nodes = neighbor_matrix[cur_node]
                mask = np.full(len(near_nodes), False, dtype=bool)
                for j in range(self.truck_num):
                    mask |= np.isin(near_nodes, routes[j])
                mask = np.invert(mask)

                unvisited_near_nodes = near_nodes[mask]
                # unvisited_near_size = np.minimum(self.neighbor_size, unvisited_near_nodes.size)
                # unvisited_near_nodes = unvisited_near_nodes[:unvisited_near_size]

                if unvisited_near_nodes.size == 0:
                    # there are cases when heuristic cannot handle zero nodes
                    cur_truck.wait()
                    continue

                next_node = eva.select_next_node(cur_node,
                                                 [r[-1] for r in routes],
                                                 destination_node,
                                                 unvisited_near_nodes,
                                                 distance_matrix)
                assert next_node in unvisited_near_nodes
                distance_left[cur_truck] = distance_matrix[cur_node][next_node]
                current_node[cur_truck] = next_node
                routes[cur_truck].append(next_node)

            # validate that we visit every node once
            flatten = [n for r in routes for n in r]
            assert len(flatten) == (self.problem_size + self.truck_num - 1)

            for i in range(self.truck_num):
                distance = self.tour_cost(instance, routes[i])
                dis[i, n_ins] = distance

            max_dis = np.max(dis[:, n_ins])
            scores.append(max_dis / ortool["max_distance"])

            results["heuristic"].append(max_dis)
            results["ortool"].append(ortool)
            results["routes"].append(routes)

            n_ins += 1

        # max_dis = np.max(dis, axis=0)
        # ave_dis = np.average(max_dis)
        # return ave_dis

        score = (np.average(scores) - 1) * 100
        return score, results


if __name__ == "__main__":
    prompts = Prompts()
    print(prompts)
