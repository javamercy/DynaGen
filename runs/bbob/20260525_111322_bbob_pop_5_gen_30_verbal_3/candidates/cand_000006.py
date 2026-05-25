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
        
        # Global exploration phase (20% of budget, at least 1)
        n_global = max(1, int(0.2 * budget))
        best_x = None
        best_val = float('inf')
        evals = 0
        for _ in range(n_global):
            if evals >= budget:
                break
            x = lb + (ub - lb) * rng.rand(dim)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        
        # Adaptive local search with restarts
        radius = 0.2  # relative to range
        stagnation = 0
        max_stagnation = max(5, int(0.1 * budget))
        while evals < budget:
            # Sample around best
            range_vec = ub - lb
            step = radius * range_vec
            x_cand = best_x + rng.uniform(-step, step)
            x_cand = np.clip(x_cand, lb, ub)
            val_cand = func(x_cand)
            evals += 1
            if evals > budget:
                break
            if val_cand < best_val:
                best_val = val_cand
                best_x = x_cand.copy()
                report_best(best_val, best_x)
                radius *= 1.2
                stagnation = 0
            else:
                stagnation += 1
                radius *= 0.95
                if radius < 1e-10:
                    radius = 0.01
                if stagnation >= max_stagnation:
                    # Restart
                    if evals < budget:
                        x_new = lb + (ub - lb) * rng.rand(dim)
                        val_new = func(x_new)
                        evals += 1
                        if val_new < best_val:
                            best_val = val_new
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                        radius = 0.2
                        stagnation = 0
                    else:
                        break
            if radius > 0.5:
                radius = 0.5
        
        # Fallback if no evaluations (should not happen)
        if evals == 0:
            x = lb + (ub - lb) * rng.rand(dim)
            best_val = func(x)
            best_x = x.copy()
            report_best(best_val, best_x)
        
        return (best_val, best_x)