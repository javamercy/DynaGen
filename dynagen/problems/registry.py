from dynagen.config import RunConfig
from dynagen.problems.base import Problem


def get_problem(problem_type: str) -> Problem:
    normalized = problem_type.lower()
    if normalized == "bbob":
        from dynagen.problems.bbob import BBOBProblem

        return BBOBProblem()
    if normalized == "dvrp":
        from dynagen.problems.dvrp import DVRPProblem

        return DVRPProblem()
    if normalized == "tsp":
        from dynagen.problems.tsp import TSPProblem

        return TSPProblem()
    raise ValueError(f"Unsupported problem type: {problem_type}")


def problem_for_config(config: RunConfig) -> Problem:
    return get_problem(config.problem.type)
