import resource
import sys
import types
import warnings
import traceback

import numpy as np

from func_timeout import func_timeout


class PromptsBase:
    def __init__(self):
        self._prompt_task: str = ""
        self._prompt_task_ext: str = ""
        self._prompt_func_name: str = ""
        self._prompt_func_inputs: list[str] = []
        self._prompt_func_outputs: list[str] = []
        self._prompt_inout_inf: str = ""
        self._prompt_other_inf: str = ""

    def __str__(self):
        return (
            f"Task:{self._prompt_task}\n"
            f"Function name: {self._prompt_func_name}\n"
            f"Function inputs: {self._prompt_func_inputs}\n"
            f"Function outputs: {self._prompt_func_outputs}\n"
            f"In-out information: {self._prompt_inout_inf}\n"
            f"Other information: {self._prompt_other_inf}"
        )

    def get_task(self):
        # Main prompt to use for initializing search
        return self._prompt_task

    def get_task_ext(self):
        # Extra prompt (want to also have conventional heuristics)
        return self._prompt_task_ext

    def get_func_name(self):
        return self._prompt_func_name

    def get_func_inputs(self):
        return self._prompt_func_inputs

    def get_func_outputs(self):
        return self._prompt_func_outputs

    def get_inout_inf(self):
        return self._prompt_inout_inf

    def get_other_inf(self):
        return self._prompt_other_inf


class ProblemBase:
    def __init__(self):
        pass

    def eval(self, algorithm):
        raise NotImplementedError

    def tour_cost(self, instance, solution):
        cost = 0
        for j in range(len(solution) - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])

        return cost

    def generate_neighborhood_matrix(self, instance):
        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            neighborhood_matrix[i] = sorted_indices

        return neighborhood_matrix

    def evaluate(self, code_string):
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
                score, results = self.eval(heuristic_module)

                # print(code_string)
                return score, results

        except Exception as e:
            # print(code_string)
            # print(traceback.format_exc())

            # print("Error:", str(e))
            return None

class GetDataBase:
    def __init__(self):
        pass

def truck_num_scaling_1(size: int) -> int:
    return size//25 + 1  # scaling 1

def truck_num_scaling(size: int) -> int:
    # programs how the number of trucks scale with the problem size
    # return size//25 + 1  # scaling 1
    # return size//30 + 2  # scaling 2
    return size//30 + 3  # scaling 3


class Requests(object):
    def __init__(self, locations, arrive_times = None, time_windows = None):
        n, _ = locations.shape

        self.locations = locations
        self.cur_time: float = 0.0

        self.incomplete = [i for i in range(1, n)]
        self.completed = [0]

        # for dynamic problems
        self.arrive_times = arrive_times
        if self.arrive_times is not None:
            self._available: list[int] = []
            self._unavailable: list[int] = []
            for i, t in enumerate(arrive_times):
                if t <= 0.0:
                    self._available.append(i)
                else:
                    self._unavailable.append(i)

        # for time window problems
        self.time_windows = time_windows
        if self.time_windows is not None:
            self.available_time = np.array([tw[0] for tw in time_windows])
            self.dead_time = np.array([tw[1] for tw in time_windows])

        # for each node get a list of its nearby nodes
        self.neighborhood_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            distances = np.linalg.norm(locations - locations[i], axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            self.neighborhood_matrix[i] = sorted_indices

    def time_step(self, step: float):
        # perform a time step
        self.cur_time += step

        # dynamic problems
        if self.arrive_times is not None:
            for i in self._unavailable:
                if self.arrive_times[i] < self.cur_time:
                    self._available.append(i)
                    self._unavailable.remove(i)

    def near_nodes(self, node: int):
        # get a list of near nodes that
        # - have arrived (i.e. not future orders)
        # - can be reached within the time window

        near_nodes = self.neighborhood_matrix[node]
        mask = np.full(near_nodes.shape, True, dtype=bool)

        if self.arrive_times is not None:
            mask &= np.isin(near_nodes, self._available)  # filter to only available nodes

        if self.time_windows is not None:
            distances = np.linalg.norm(self.locations[near_nodes] - self.locations[node], axis=1)

            # available_times = np.array([self.available_time[n] for n in near_nodes])
            # mask &= self.cur_time + distances >= available_times  # filter to nodes that can be reached after they are available

            dead_times = np.array([self.dead_time[n] for n in near_nodes])
            mask &= (self.cur_time + distances) <= dead_times  # filter to nodes that can be reached before they are unavailable

        return near_nodes[mask]

    def currently_at(self, node: int, from_t: float, to_t: float):
        assert from_t <= to_t
        if node not in self.incomplete:
            return

        mark_done = True

        # dynamic problems
        if self.arrive_times is not None:
            if to_t < self.arrive_times[node]:
                mark_done = False

        # time window problems
        if self.time_windows is not None:
            if to_t < self.available_time[node]:
                mark_done = False
            if self.dead_time[node] < from_t:
                mark_done = False

        if mark_done:
            self.incomplete.remove(node)
            self.completed.append(node)


    def is_done(self) -> bool:
        if self.cur_time > 1000:
            return True

        if len(self.incomplete) == 0:
            return True

        if self.time_windows is not None:
            if self.cur_time > max([self.time_windows[n][1] for n in self.incomplete]):
                return True

        return False


class Truck(object):
    def __init__(self, requests: Requests):
        self.route: list[int] = [0]
        self._time_moving: float = 0.0
        self._time_waited: float = 0.0

        self.requests = requests
        self.locations = np.array(requests.locations)
        self.depot = self.locations[0]
        self.cur_time: float = 0.0

        # _pos: the current position of the truck
        # _dest: the destination currently being traveled to (None if we are there)
        # _wait_left: time to wait a position
        self._pos: np.array = np.array(self.depot)
        self._dest: np.array = None
        self._wait_left: float = 0.0

    def set_dest(self, node: int):
        # set a destination, make sure we are not traveling or waiting
        assert self._dest is None
        assert self._wait_left == 0.0

        self.requests.currently_at(self.cur_node(), self.cur_time, self.cur_time)
        if self.cur_node() == node:
            self.wait()
        else:
            self.route.append(node)
            self._dest = self.locations[node]

    def time_left(self) -> float:
        # get the time left to reach destination or finish waiting
        if self._dest is None:
            return self._wait_left
        return np.linalg.norm(self._pos - self._dest)

    def cur_node(self) -> int:
        # return the current node we are at, make sure we are not traveling
        assert self._dest is None
        return self.route[-1]

    def wait(self):
        # wait at the current node, make sure we are not traveling
        assert self._dest is None
        self._wait_left = 0.1

    def time_step(self, step: float):
        # perform a time step
        self.cur_time += step

        if self._dest is None:
            # if we are not traveling update the wait time left
            assert step <= self._wait_left
            self.requests.currently_at(self.cur_node(), self.cur_time - step, self.cur_time)
            self._wait_left -= step
            self._time_waited += step
            if self._wait_left <= 1e-8:
                self._wait_left = 0.0

        else:
            # if we are traveling move the truck closer to the destination
            want = np.linalg.norm(self._pos - self._dest)
            assert step <= want
            self._pos += (self._dest - self._pos) * (step/want)
            self._time_moving += step
            if abs(step - want) <= 1e-8:
                self._pos = np.array(self._dest)
                self._dest = None
                self.requests.currently_at(self.cur_node(), self.cur_time, self.cur_time)

    def go_home(self):
        # finish the current task (get to location or finish waiting) then go to depot
        if self._dest is None:
            # if we are not traveling use up all the wait time left
            self.cur_time += self._wait_left
            self.requests.currently_at(self.cur_node(), self.cur_time - self._wait_left, self.cur_time)

            self._time_waited += self._wait_left
            self._wait_left = 0.0

        else:
            # if we are traveling move the truck to its destination
            dist = np.linalg.norm(self._pos - self._dest)
            assert self._wait_left == 0.0

            self.cur_time += dist

            self._pos = np.array(self._dest)
            self._time_moving += dist
            self._dest = None
            self.requests.currently_at(self.cur_node(), self.cur_time, self.cur_time)

        self.route.append(0)
        self._time_moving += np.linalg.norm(self._pos - self.depot)
        self._pos = np.array(self.depot)

    def tour_cost(self, solution):
        cost = 0
        for j in range(len(solution) - 1):
            cost += np.linalg.norm(self.locations[int(solution[j])] - self.locations[int(solution[j + 1])])
        cost += np.linalg.norm(self.locations[int(solution[-1])] - self.locations[int(solution[0])])

        if abs(cost - self._time_moving) >= 1e-8:
            print(f"Error: {cost} vs {self._time_moving}")

        return self._time_moving + self._time_waited
