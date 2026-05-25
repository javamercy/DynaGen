import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial point
        x = lb + rng.rand(dim) * (ub - lb)
        val = func(x)
        evals = 1
        best_x = x.copy()
        best_val = val
        report_best(best_val, best_x)

        # Simulated annealing parameters
        T0 = 1.0
        T = T0
        cooling_rate = 0.95
        sigma = 0.1 * (ub - lb) / np.sqrt(dim)

        k = 0
        while evals < budget:
            # Generate candidate by Gaussian perturbation
            perturbation = rng.normal(0, sigma, dim)
            candidate = x + perturbation
            candidate = np.clip(candidate, lb, ub)
            cand_val = func(candidate)
            evals += 1

            delta = cand_val - val
            # Accept if improvement or with probability
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                x = candidate
                val = cand_val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

            # Update temperature
            T = T0 * (cooling_rate ** k)
            k += 1

        return best_val, best_x