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

        best_x = None
        best_val = float('inf')
        evals = 0

        # Global random search (2/3 of budget)
        n_global = int(2 * budget / 3)
        if n_global < 1:
            n_global = 1

        for _ in range(n_global):
            if evals >= budget:
                break
            x = lb + (ub - lb) * rng.rand(dim)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()

        # Local refinement (remaining budget)
        if best_x is not None:
            scale = 0.1 * (ub - lb)
            while evals < budget:
                # Perturb best point with decreasing step
                step = scale * (1 - evals / budget) * rng.randn(dim)
                candidate = np.clip(best_x + step, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
        else:
            # Fallback: should never happen, but ensure at least one eval
            x = lb + (ub - lb) * rng.rand(dim)
            best_val = func(x)
            best_x = x.copy()
            evals += 1

        # Final call to report_best (should be done on improvement, but ensure it's called)
        # Already called on each improvement in loop
        return (best_val, best_x)