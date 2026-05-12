import numpy as np
from src.eoh.problems.interface import PromptsBase, ProblemBase, Truck, Requests, truck_num_scaling

TRUCK_SPEED = 1.0


class Prompts(PromptsBase):
    def __init__(self):
        super().__init__()
        self._prompt_task: str = (
            "With a fleet of trucks, you are tasked with finding routes to visit a list of customers and return to the start. "
            "This task can be solved step-by-step by iteratively choosing the next node every time a truck arrives at a node. "
            "However, some customers have a time window they are available in and will incur a large penalty if missed and over the course of the task more customers will be added. "
            "There is also the option for the truck to wait at a node for a period of time if 'None' is returned. "
            "The overall goal is to minimize the time it takes for the last truck to finish. "
            "Help me design a novel algorithm that is different from the algorithms in literature."
        )
        self._prompt_task_ext: str = (
            "With a fleet of trucks, you are tasked with finding routes to visit a list of customers and return to the start. "
            "This task can be solved step-by-step by iteratively choosing the next node every time a truck arrives at a node. "
            "However, some customers have a time window they are available in and will incur a large penalty if missed and over the course of the task more customers will be added. "
            "There is also the option for the truck to wait at a node for a period of time if 'None' is returned. "
            "The overall goal is to minimize the time it takes for the last truck to finish. "
        )

        self._prompt_func_name: str = "select_next_node"
        self._prompt_func_inputs: list[str] = ["cur_truck_pos",
                                               "depot_pos",
                                               "all_truck_pos",
                                               "unvisited_customers",
                                               "current_time",
                                               "truck_speed",
                                               "time_windows"]
        self._prompt_func_outputs: list[str] = ["best_node"]
        self._prompt_inout_inf: str = (
            "'cur_truck_pos' are 'depot_pos' both coordinates. "
            "'all_truck_pos' and 'unvisited_customers' are both lists of coordinates. "
            "'current_time' and 'truck_speed' are floats and 'time_windows' is a list of pairs of floats. "
            "The function must end with 'return best_node' which should contain an index of the 'unvisited' list to indicate where the current truck should go. "
        )
        self._prompt_other_inf: str = (
            "Coordinates are a numpy array of an x and y position"
        )


class DVRPTWConstruct(ProblemBase):
    def __init__(self, data, size: int, n_test: int):
        super().__init__()
        self.ndelay = 1
        self.neighbor_size = np.minimum(50, size)
        self.running_time = 10

        self.instance_data = data
        self.problem_size = size
        self.n_instance = n_test
        self.truck_num = truck_num_scaling(size)

        self.prompts: PromptsBase = Prompts()


    def eval(self, eva):
        dis = np.zeros((self.truck_num, self.n_instance))
        results = {
            "heuristic": [],
            "ortool": [],
            "routes": [],
            "missed": [],
        }
        scores = []

        for i, (instance, arrive_times, time_windows, ortool) in enumerate(self.instance_data):
            if i == self.n_instance:
                break

            last_pos = instance[0]
            request_handler = Requests(instance,
                                       arrive_times=arrive_times,
                                       time_windows=time_windows)
            trucks = [Truck(request_handler) for _ in range(self.truck_num)]

            while True:
                truck = trucks[np.argmin([truck.time_left() for truck in trucks])]
                t_left = truck.time_left()
                if t_left > 0.0:
                    for t in trucks:
                        t.time_step(t_left)
                    request_handler.time_step(t_left)

                if request_handler.is_done():
                    break

                near_nodes = request_handler.near_nodes(truck.cur_node())
                mask = np.isin(near_nodes, request_handler.completed)
                for t in trucks:
                    if t is not truck:
                        mask |= np.isin(near_nodes, t.route[-1])
                mask = np.invert(mask)

                unvisited_near_nodes = near_nodes[mask]
                unvisited_near_size = np.minimum(self.neighbor_size, unvisited_near_nodes.size)
                unvisited_near_nodes = unvisited_near_nodes[:unvisited_near_size]

                if unvisited_near_nodes.size == 0:
                    # there are cases when heuristic cannot handle zero nodes
                    truck.wait()
                    continue

                next_node_idx = eva.select_next_node(np.array(truck._pos),
                                                     np.array(last_pos),
                                                     [np.array(t._pos) for t in trucks],
                                                     [instance[node].copy() for node in unvisited_near_nodes],
                                                     request_handler.cur_time,
                                                     TRUCK_SPEED,
                                                     [time_windows[node] for node in unvisited_near_nodes])

                if next_node_idx is None:
                    truck.wait()
                else:
                    assert next_node_idx >= 0
                    assert next_node_idx < unvisited_near_nodes.size
                    next_node = unvisited_near_nodes[next_node_idx]
                    # print(f"Time {request_handler.cur_time} at {next_node} ({time_windows[next_node][0]}, {time_windows[next_node][1]})")
                    truck.set_dest(next_node)

            for t in trucks:
                t.go_home()

            flattened = set([n for t in trucks for n in t.route])
            missed = self.problem_size - len(flattened)

            for j, t in enumerate(trucks):
                distance = t.tour_cost(t.route)
                dis[j, i] = distance

            max_dis = np.max(dis[:, i])
            results["heuristic"].append(max_dis)
            results["ortool"].append(ortool)
            results["routes"].append([t.route for t in trucks])
            results["missed"].append(missed)

            max_dis = np.max(dis[:, i]) + (missed - ortool["missed"]) * 2.5
            scores.append(max_dis / ortool["max_distance"])

        # max_dis = np.max(dis, axis=0)
        # ave_dis = np.average(max_dis)
        # return ave_dis

        score = (np.average(scores) - 1) * 100
        return score, results


if __name__ == "__main__":
    prompts = Prompts()
    print(prompts)
