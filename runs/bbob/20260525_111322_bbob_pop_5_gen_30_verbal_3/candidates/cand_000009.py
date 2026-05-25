import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        best_val = float('inf')
        best_x = None

        while budget > 0:
            # Start a new pattern search from random point
            x = lb + (ub - lb) * rng.rand(dim)
            val = func(x)
            budget -= 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

            step = 0.1 * np.mean(ub - lb)
            step_min = 1e-10 * np.mean(ub - lb)

            # Generate directions: positive and negative axes
            directions = []
            for i in range(dim):
                e = np.zeros(dim)
                e[i] = 1.0
                directions.append(e)
                directions.append(-e)

            improved = True
            while budget > 0 and step > step_min:
                if not improved:
                    step *= 0.5
                    if step < step_min:
                        break
                improved = False
                for d in directions:
                    if budget <= 0:
                        break
                    candidate = np.clip(x + step * d, lb, ub)
                    val = func(candidate)
                    budget -= 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        x = candidate
                        improved = True
                        break
                if not improved:
                    step *= 0.5
        return best_val, best_x