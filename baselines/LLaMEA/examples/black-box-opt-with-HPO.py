# This is a more advanced example of how to use the LLaMEA algorithm with the Gemini LLM to generate optimization algorithms for the BBOB test suite.
# It includes the in-the-loop hyper-parameter optimization (HPO) using SMAC extension of LLaMEA and a more complex evaluation function.
# We have to define the following components for LLaMEA to work:
# - An evaluation function that executes the generated code and evaluates its performance.
# - A task prompt that describes the problem to be solved.
# - An LLM instance that will generate the code based on the task prompt.


import os
import re
import textwrap
import time
from itertools import product
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ioh import get_problem, logger
from smac import AlgorithmConfigurationFacade, Scenario

from llamea import Gemini_LLM, LLaMEA, Ollama_LLM, OpenAI_LLM
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
    print(f"LLaMEA-HPO output directory: {output_dir}")


def _build_llm(model: str):
    provider = os.getenv("LLAMEA_LLM_PROVIDER", "gemini").lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAI_LLM(api_key, model)
    if provider == "ollama":
        return Ollama_LLM(model)
    api_key = os.getenv("GOOGLE_API_KEY")
    return Gemini_LLM(api_key, model)


def _parse_instance(instance):
    if isinstance(instance, tuple):
        return int(instance[0]), int(instance[1])
    fid, iid = str(instance).strip().strip("()").split(",", 1)
    return int(fid.strip()), int(iid.strip())

if __name__ == "__main__":
    # Execution code starts here
    ai_model = os.getenv("LLM_MODEL", os.getenv("LLAMEA_LLM_MODEL", "gemini-1.5-flash"))
    experiment_name = os.getenv("LLAMEA_EXPERIMENT_NAME", "pop1-5-HPO")
    bbob_dimensions = _env_int_list("LLAMEA_BBOB_DIMENSIONS", [5])
    bbob_function_ids = _env_int_list("LLAMEA_BBOB_FUNCTION_IDS", range(1, 25))
    bbob_instance_ids = _env_int_list("LLAMEA_BBOB_INSTANCE_IDS", [1, 2, 3])
    bbob_repetitions = _env_int("LLAMEA_BBOB_REPETITIONS", 3)
    bbob_budget_factor = _env_int("LLAMEA_BBOB_BUDGET_FACTOR", 2000)
    llm_call_budget = _env_int("LLAMEA_LLM_CALLS", 100)
    n_parents = _env_int("LLAMEA_N_PARENTS", 5)
    n_offspring = _env_int("LLAMEA_N_OFFSPRING", 5)
    eval_timeout = _env_int("LLAMEA_EVAL_TIMEOUT", 3600)
    hpo_trials = _env_int("LLAMEA_HPO_TRIALS", 2000)
    hpo_min_budget = _env_int("LLAMEA_HPO_MIN_BUDGET", 12)
    hpo_max_budget = _env_int("LLAMEA_HPO_MAX_BUDGET", 200)
    _prepare_output_dir()
    llm = _build_llm(ai_model)

    def evaluateBBOBWithHPO(solution, explogger=None):
        """
        Evaluates an optimization algorithm on the BBOB (Black-Box Optimization Benchmarking) suite and computes
        the Area Over the Convergence Curve (AOCC) to measure performance. In addddition, if a configuration space is provided, it
        applies Hyper-parameter optimization with SMAC first.

        Parameters:
        -----------
        solution : dict
            A dictionary containing "_solution" (the code to evaluate), "_name", "_description" and "_configspace"

        explogger : logger
            A class to log additional stuff for the experiment.

        Returns:
        --------
        solution : dict
            Updated solution with "_fitness", "_feedback", "incumbent" and optional "_error"

        Functionality:
        --------------
        - Executes the provided `code` string in the global context, allowing for dynamic inclusion of necessary components.
        - Iterates over a predefined set of dimensions (currently only 5), function IDs (1 to 24), and instance IDs (1 to 3).
        - For each problem, the specified algorithm is instantiated and executed with a defined budget.
        - AOCC is computed for each run, and the results are aggregated across all runs, problems, and repetitions.
        - The function handles cases where the algorithm exceeds its budget using an `OverBudgetException`.
        - Logs the results if an `explogger` is provided.
        - The function returns a feedback string, the mean AOCC score, and an error placeholder.

        Notes:
        ------
        - The budget for each algorithm run is set to 10,000.
        - The function currently only evaluates a single dimension (5), but this can be extended.
        - Hyperparameter Optimization (HPO) with SMAC is mentioned but not implemented.
        - The AOCC score is a metric where 1.0 is the best possible outcome, indicating optimal convergence.

        """
        auc_mean = 0
        auc_std = 0
        code = solution.code
        algorithm_name = solution.name
        exec(code, globals())
        dim = bbob_dimensions[0]
        budget = bbob_budget_factor * dim
        error = ""
        algorithm = None

        # perform a small run to check for any code errors
        l2_temp = aoc_logger(100, upper=1e2, triggers=[logger.trigger.ALWAYS])
        problem = get_problem(11, 1, dim)
        problem.attach_logger(l2_temp)
        try:
            algorithm = globals()[algorithm_name](budget=100, dim=dim)
            algorithm(problem)
        except OverBudgetException:
            pass

        # now optimize the hyper-parameters
        def get_bbob_performance(config: Configuration, instance: str, seed: int = 0):
            np.random.seed(seed)
            fid, iid = _parse_instance(instance)
            problem = get_problem(fid, iid, dim)
            l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            problem.attach_logger(l2)
            try:
                algorithm = globals()[algorithm_name](
                    budget=budget, dim=dim, **dict(config)
                )
                algorithm(problem)
            except OverBudgetException:
                pass
            except Exception as e:
                print(problem.state, budget, e)
            auc = correct_aoc(problem, l2, budget)
            return 1 - auc

        args = list(product(bbob_function_ids, bbob_instance_ids))
        np.random.shuffle(args)
        inst_feats = {str(arg): [arg[0]] for idx, arg in enumerate(args)}
        # inst_feats = {str(arg): [idx] for idx, arg in enumerate(args)}
        error = ""

        if solution.configspace is None:
            # No HPO possible, evaluate only the default
            incumbent = {}
            error = "The configuration space was not properly formatted or not present in your answer. The evaluation was done on the default configuration."
        else:
            configuration_space = solution.configspace
            scenario = Scenario(
                configuration_space,
                name=str(int(time.time())) + "-" + algorithm_name,
                deterministic=False,
                min_budget=hpo_min_budget,
                max_budget=hpo_max_budget,
                n_trials=hpo_trials,
                instances=args,
                instance_features=inst_feats,
                output_directory="smac3_output"
                if explogger is None
                else explogger.dirname + "/smac"
                # n_workers=10
            )
            smac = AlgorithmConfigurationFacade(
                scenario, get_bbob_performance, logging_level=30
            )
            incumbent = smac.optimize()

        # last but not least, perform the final validation

        aucs = []
        for validation_dim in bbob_dimensions:
            validation_budget = bbob_budget_factor * validation_dim
            l2 = aoc_logger(validation_budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            for fid in bbob_function_ids:
                for iid in bbob_instance_ids:
                    problem = get_problem(fid, iid, validation_dim)
                    problem.attach_logger(l2)
                    for rep in range(bbob_repetitions):
                        np.random.seed(rep)
                        try:
                            algorithm = globals()[algorithm_name](
                                budget=validation_budget, dim=validation_dim, **dict(incumbent)
                            )
                            algorithm(problem)
                        except OverBudgetException:
                            pass
                        auc = correct_aoc(problem, l2, validation_budget)
                        aucs.append(auc)
                        l2.reset(problem)
                        problem.reset()

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        dict_hyperparams = dict(incumbent)
        feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with optimal hyperparameters {dict_hyperparams}."
        print(algorithm_name, algorithm, auc_mean, auc_std)

        solution.add_metadata("aucs", aucs)
        solution.add_metadata("incumbent", dict_hyperparams)
        solution.set_scores(auc_mean, feedback)

        return solution

    role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
    task_prompt = textwrap.dedent("""
    The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
    The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

    In addition, any hyper-parameters the algorithm uses will be optimized by SMAC, for this, provide a Configuration space as Python dictionary (without the dim and budget parameters) and include all hyper-parameters in the __init__ function header.
    An example configuration space is as follows:

    ```python
    {
        "float_parameter": (0.1, 1.5),
        "int_parameter": (2, 10),
        "categoral_parameter": ["mouse", "cat", "dog"]
    }
    ```

    Give an excellent and novel heuristic algorithm including its configuration space to solve this task and also give it a one-line description, describing the main idea.
    """)

    format_prompt = textwrap.dedent("""
    Give the response in the format:
    # Description: <short-description>
    # Code: <code>
    # Space: <configuration_space>
    """)

    example_prompt = textwrap.dedent("""
    An example of such code (a simple random search), is as follows:
    ```python
    import numpy as np

    class RandomSearch:
        def __init__(self, budget=10000, dim=10):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            self.f_opt = np.inf
            self.x_opt = None
            for i in range(self.budget):
                x = np.random.uniform(func.bounds.lb, func.bounds.ub)

                f = func(x)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x

            return self.f_opt, self.x_opt
    ```
    """)

    feedback_prompts = [
        f"Either refine or redesign to improve the solution (and give it a distinct one-line description)."
    ]

    for experiment_i in [1]:
        es = LLaMEA(
            evaluateBBOBWithHPO,
            llm=llm,
            n_parents=n_parents,
            n_offspring=n_offspring,
            role_prompt=role_prompt,
            task_prompt=task_prompt,
            example_prompt=example_prompt,
            output_format_prompt=format_prompt,
            mutation_prompts=feedback_prompts,
            experiment_name=experiment_name,
            elitism=True,
            HPO=True,
            budget=llm_call_budget,
            eval_timeout=eval_timeout,
        )
        result = es.run()
        if getattr(es, "logger", None) is not None:
            result.add_metadata("llm_model", es.model)
        print(result)
