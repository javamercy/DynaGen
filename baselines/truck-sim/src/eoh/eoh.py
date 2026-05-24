import random

from .utils import createFolders
from .evolution import methods
from .problems import problems


class EVOL:
    def __init__(self, paras, prob=None):
        print("-----------------------------------------\n"
              "---              Start EoH            ---\n"
              "-----------------------------------------")

        createFolders.create_folders(paras.exp_output_path)
        print("- output folder created -")

        self.paras = paras
        print("-  parameters loaded -")

        self.prob = prob
        random.seed(2024)  # set a random seed

    def run(self, instances, size: int, n_test: int):
        problem_generator = problems.Probs(self.paras.problem)
        problem = problem_generator.get_problem(instances, size, n_test)
        print(f"- problem {self.paras.problem} loaded -")

        method_generator = methods.Methods(self.paras, problem)
        method = method_generator.get_method()

        method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------\n"
              "---     EoH successfully finished !   ---\n"
              "-----------------------------------------")
