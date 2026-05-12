from .interface import ProblemBase

from joblib import Parallel, delayed


def parallel_get_data(data_getter, n_test: int, n_threads: int):
    data = []

    while True:
        instances = Parallel(n_jobs=n_threads)(
            delayed(data_getter.generate_instance)() for _ in range(n_threads))

        for instance in instances:
            if len(data) == n_test:
                break

            if instance[-1] is not None and instance[-2] is not None:
                data.append(instance)
                print(".", end="", flush=True)
            else:
                print("x", end="", flush=True)

        if len(data) == n_test:
            break

    print()
    return data


class Probs:
    def __init__(self, problem: str):
        self.problem = problem

    def get_data(self, size: int, n_test: int, n_threads: int = 1):
        if not isinstance(self.problem, str):
            prob = self.problem
            print("- Cannot generate data for unknown problem! ")

        else:
            print(f"Generating {n_test} instances of size {size}: ", end="", flush=True)

            match self.problem:
                case "tsp_construct":
                    from .tsp_construct.data import GetData
                case "tsp_gls":
                    from .tsp_construct.data import GetData
                case "vrp_construct":
                    from .vrp_construct.data import GetData
                case "vrp_gls":
                    from .vrp_construct.data import GetData
                case "dvrp_construct":
                    from .dvrp_construct.data import GetData
                case "vrptw_construct":
                    from .vrptw_construct.data import GetData
                case "dvrptw_construct":
                    from .dvrptw_construct.data import GetData
                case _:
                    raise Exception(f"Problem {self.problem} not found!")

            return parallel_get_data(GetData(size), n_test, n_threads=n_threads)


    def get_problem(self, instances, size: int, n_test: int) -> ProblemBase:
        if not isinstance(self.problem, str):
            prob = self.problem
            # print("- Prob local loaded ")

        else:
            match self.problem:
                case "tsp_construct":
                    from .tsp_construct import TSPConstruct
                    prob = TSPConstruct(instances, size, n_test)
                case "tsp_gls":
                    from .tsp_gls import TSPGLS
                    prob = TSPGLS(instances, size, n_test)
                case "vrp_construct":
                    from .vrp_construct import VRPConstruct
                    prob = VRPConstruct(instances, size, n_test)
                case "vrp_gls":
                    from .vrp_gls import VRPGLS
                    prob = VRPGLS(instances, size, n_test)
                case "dvrp_construct":
                    from .dvrp_construct import DVRPConstruct
                    prob = DVRPConstruct(instances, size, n_test)
                case "vrptw_construct":
                    from .vrptw_construct import VRPTWConstruct
                    prob = VRPTWConstruct(instances, size, n_test)
                case "dvrptw_construct":
                    from .dvrptw_construct import DVRPTWConstruct
                    prob = DVRPTWConstruct(instances, size, n_test)

                case _:
                    raise Exception(f"Problem {self.problem} not found!")

            # print(f"- problem {self.problem} loaded ")
        return prob
