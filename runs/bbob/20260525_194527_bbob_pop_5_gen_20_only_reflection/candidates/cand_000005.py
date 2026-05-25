import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        best_y = np.inf
        best_x = None
        evals = 0

        # Initial population: random sampling (global phase)
        n_init = max(1, min(budget // 2, 10 * dim))
        for _ in range(n_init):
            if evals >= budget:
                break
            x = np.random.uniform(lb, ub, size=dim)
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
                report_best(best_y, best_x)

        # If we have no best yet (should not happen as we have at least one eval)
        if best_x is None:
            # evaluate a random point
            x = np.random.uniform(lb, ub, size=dim)
            best_y = func(x)
            evals += 1
            best_x = x.copy()
            report_best(best_y, best_x)

        # Local refinement with (1+1)-ES
        # Adaptive step sizes per dimension
        sigma = 0.2 * (ub - lb)
        # To avoid too small steps
        min_sigma = 1e-10 * (ub - lb)
        while evals < budget:
            # generate candidate
            noise = np.random.randn(dim)
            x_candidate = best_x + sigma * noise
            # clip to bounds
            x_candidate = np.clip(x_candidate, lb, ub)
            y_candidate = func(x_candidate)
            evals += 1
            if y_candidate < best_y:
                best_y = y_candidate
                best_x = x_candidate.copy()
                report_best(best_y, best_x)
                # increase sigma (exploitation)
                sigma = sigma * 1.2
            else:
                # decrease sigma (exploration)
                sigma = sigma * 0.85
            # ensure sigma does not become too small
            sigma = np.maximum(sigma, min_sigma)

        return best_y, best_x