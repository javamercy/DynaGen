# This is a minimal example of how to use the LLaMEA algorithm with the Gemini LLM to generate optimization algorithms for the BBOB test suite.
# We have to define the following components for LLaMEA to work:
# - An evaluation function that executes the generated code and evaluates its performance.
# - A task prompt that describes the problem to be solved.
# - An LLM instance that will generate the code based on the task prompt.

import os
import pickle
import textwrap
from pathlib import Path

import numpy as np
from ioh import get_problem, logger

from llamea import Gemini_LLM, LLaMEA, Ollama_LLM, OpenAI_LLM
from llamea.utils import prepare_namespace, clean_local_namespace
from misc import OverBudgetException, aoc_logger, correct_aoc

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_int_list(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _prepare_output_dir():
    output_dir = Path(os.getenv("LLAMEA_OUTPUT_DIR", str(PROJECT_ROOT / "runs" / "bbob")))
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    print(f"LLaMEA output directory: {output_dir}")


def _build_llm(model: str):
    provider = "openai"
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAI_LLM(api_key, model)
    if provider == "ollama":
        return Ollama_LLM(model)
    api_key = os.getenv("GOOGLE_API_KEY")
    return Gemini_LLM(api_key, model)


if __name__ == "__main__":
    # Execution code starts here
    ai_model = os.getenv("LLM_MODEL", os.getenv("LLAMEA_LLM_MODEL", "gpt-5.4-nano"))
    experiment_name = os.getenv("LLAMEA_EXPERIMENT_NAME", "pop1-5")
    bbob_dimensions = _env_int_list("LLAMEA_BBOB_DIMENSIONS", [5])
    bbob_function_ids = _env_int_list("LLAMEA_BBOB_FUNCTION_IDS", range(1, 25))
    bbob_instance_ids = _env_int_list("LLAMEA_BBOB_INSTANCE_IDS", [1, 2, 3])
    bbob_repetitions = _env_int("LLAMEA_BBOB_REPETITIONS", 3)
    bbob_budget_factor = _env_int("LLAMEA_BBOB_BUDGET_FACTOR", 2000)
    llm_call_budget = _env_int("LLAMEA_LLM_CALLS", 100)
    n_parents = _env_int("LLAMEA_N_PARENTS", 1)
    n_offspring = _env_int("LLAMEA_N_OFFSPRING", 1)
    _prepare_output_dir()
    llm = _build_llm(ai_model)


    # We define the evaluation function that executes the generated algorithm (solution.code) on the BBOB test suite.
    # It should set the scores and feedback of the solution based on the performance metric, in this case we use mean AOCC.
    def evaluateBBOB(solution, explogger=None):
        auc_mean = 0
        auc_std = 0

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

        aucs = []

        algorithm = None
        for dim in bbob_dimensions:
            budget = bbob_budget_factor * dim
            l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            for fid in bbob_function_ids:
                for iid in bbob_instance_ids:
                    problem = get_problem(fid, iid, dim)
                    problem.attach_logger(l2)

                    for rep in range(bbob_repetitions):
                        np.random.seed(rep)
                        try:
                            algorithm = local_ns[algorithm_name](
                                budget=budget, dim=dim
                            )
                            algorithm(problem)
                        except OverBudgetException:
                            pass

                        auc = correct_aoc(problem, l2, budget)
                        aucs.append(auc)
                        l2.reset(problem)
                        problem.reset()
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)

        feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard deviation {auc_std:0.4f}."

        print(algorithm_name, algorithm, auc_mean, auc_std)
        solution.add_metadata("aucs", aucs)
        solution.set_scores(auc_mean, feedback)

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
            n_parents=n_parents,
            n_offspring=n_offspring,
            llm=llm,
            task_prompt=task_prompt,
            experiment_name=experiment_name,
            elitism=True,
            HPO=False,
            budget=llm_call_budget
        )
        result = es.run()
        if getattr(es, "logger", None) is not None:
            result.add_metadata("llm_model", es.model)
        print(result)
