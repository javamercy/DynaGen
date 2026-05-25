import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        # Initial temperature: set to range of function values? We'll use a heuristic.
        # First, evaluate a random point to get initial value.
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_val = func(best_x)
        report_best(best_val, best_x)
        evals = 1

        current_x = best_x.copy()
        current_val = best_val
        T = 1.0  # initial temperature, will be adjusted
        cooling_rate = 0.95
        restart_patience = max(10, dim)  # number of iterations without improvement before restart
        no_improve_count = 0

        while evals < budget:
            # Generate candidate by Gaussian perturbation scaled by T
            # Use step size proportional to bounds range
            step_size = 0.1 * (ub - lb) * T / (1.0 + T)  # adapt step size to temperature
            candidate = current_x + rng.normal(0, step_size, dim)
            candidate = np.clip(candidate, lb, ub)

            candidate_val = func(candidate)
            evals += 1

            delta = candidate_val - current_val
            if delta < 0:
                # Better: always accept
                current_x = candidate.copy()
                current_val = candidate_val
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                no_improve_count = 0
            else:
                # Worse: accept with probability exp(-delta/T)
                if rng.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_x = candidate.copy()
                    current_val = candidate_val
                no_improve_count += 1

            # Cool temperature
            T *= cooling_rate

            # Check for restart
            if no_improve_count >= restart_patience and evals < budget:
                # Restart from a new random point
                current_x = lb + rng.rand(dim) * (ub - lb)
                current_val = func(current_x)
                evals += 1
                if current_val < best_val:
                    best_val = current_val
                    best_x = current_x.copy()
                    report_best(best_val, best_x)
                no_improve_count = 0
                # Reset temperature to initial value to encourage exploration
                T = 1.0

        return best_val, best_x