# This is a minimal example of how to use the LLaMEA algorithm with the Gemini LLM to generate optimization algorithms for the BBOB test suite.
# We have to define the following components for LLaMEA to work:
# - An evaluation function that executes the generated code and evaluates its performance.
# - A task prompt that describes the problem to be solved.
# - An LLM instance that will generate the code based on the task prompt.

import os
import pickle
import textwrap
from datetime import datetime

import numpy as np
from ioh import get_problem, logger

from llamea import LLaMEA
from llamea.llm import DeepSeek_LLM
from llamea.utils import prepare_namespace, clean_local_namespace
from misc import OverBudgetException, aoc_logger, correct_aoc

if __name__ == "__main__":
    # Execution code starts here
    api_key = os.getenv("DEEPSEEK_API_KEY")
    ai_model = "deepseek-v4-flash"
    llm = DeepSeek_LLM(api_key, model=ai_model)
    n_gens = 5
    experiment_name = f"bbob-{n_gens}_gens"


    # We define the evaluation function that executes the generated algorithm (solution.code) on the BBOB test suite.
    # It evaluates on train instances [1,2,3] for feedback to LLM, and on test instances [4,5] for held-out evaluation.
    def evaluateBBOB(solution, explogger=None):
        code = solution.code
        algorithm_name = solution.name
        feedback = ""
        possible_issue = None
        local_ns = {}
        try:
            global_ns, possible_issue = prepare_namespace(code, allowed=["numpy"], logger=explogger)
            exec(code, global_ns, local_ns)
            local_ns = clean_local_namespace(local_ns, global_ns)

        except Exception as e:
            if possible_issue:
                feedback = f" {possible_issue}."
            solution.set_scores(float("-inf"), feedback, e)
            return solution

        train_aucs = []
        test_aucs = []

        algorithm = None
        for dim in [5]:
            budget = 2000 * dim
            l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            for fid in np.arange(1, 25):
                # Train instances (used for LLM feedback)
                for iid in [1, 2, 3]:
                    problem = get_problem(fid, iid, dim)
                    problem.attach_logger(l2)

                    for rep in range(3):
                        np.random.seed(rep)
                        try:
                            algorithm = local_ns[algorithm_name](
                                budget=budget, dim=dim
                            )
                            algorithm(problem)
                        except OverBudgetException:
                            pass

                        auc = correct_aoc(problem, l2, budget)
                        train_aucs.append(auc)
                        l2.reset(problem)
                        problem.reset()

                # Test instances (held-out evaluation)
                for iid in [4, 5]:
                    problem = get_problem(fid, iid, dim)
                    problem.attach_logger(l2)

                    for rep in range(3):
                        np.random.seed(rep)
                        try:
                            algorithm = local_ns[algorithm_name](
                                budget=budget, dim=dim
                            )
                            algorithm(problem)
                        except OverBudgetException:
                            pass

                        auc = correct_aoc(problem, l2, budget)
                        test_aucs.append(auc)
                        l2.reset(problem)
                        problem.reset()

        train_auc_mean = np.mean(train_aucs)
        train_auc_std = np.std(train_aucs)
        test_auc_mean = np.mean(test_aucs)
        test_auc_std = np.std(test_aucs)

        feedback = f"The algorithm {algorithm_name} got an average AOCC score of {train_auc_mean:0.4f} (train) / {test_auc_mean:0.4f} (test)."

        print(algorithm_name, algorithm, train_auc_mean, test_auc_mean)
        solution.add_metadata("train_aucs", train_aucs)
        solution.add_metadata("test_aucs", test_aucs)
        solution.add_metadata("train_fitness", train_auc_mean)
        solution.add_metadata("test_fitness", test_auc_mean)
        solution.set_scores(train_auc_mean, feedback)

        return solution


    # The task prompt describes the problem to be solved by the LLaMEA algorithm.
    task_prompt = textwrap.dedent("""
    The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
    The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
    Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea.
    """)

    for experiment_i in [1]:
        # A 1+1 strategy
        es = LLaMEA(
            evaluateBBOB,
            n_parents=1,
            n_offspring=1,
            llm=llm,
            task_prompt=task_prompt,
            experiment_name=experiment_name,
            elitism=True,
            HPO=False,
            budget=n_gens,
            eval_timeout=180
        )
        print(es.run())
