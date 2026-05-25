import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget

        def clip(x):
            return np.clip(x, lb, ub)

        # Initialization
        x = rng.uniform(lb, ub, size=dim)
        f = func(x)
        budget -= 1
        best_x = x.copy()
        best_f = f
        report_best(best_f, best_x)

        # SA parameters
        T = 1.0
        cooling_rate = 0.99
        sigma = 0.2 * (ub - lb)  # step size per dimension

        while budget > 0:
            # Propose candidate
            candidate = x + sigma * rng.randn(dim)
            candidate = clip(candidate)
            candidate_f = func(candidate)
            budget -= 1

            delta = candidate_f - f
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                x = candidate
                f = candidate_f
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)

            # Cool down
            T *= cooling_rate

        return best_f, best_x