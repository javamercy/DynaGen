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
        
        # Per-dimension step sizes initialized to 20% of range
        step_sizes = 0.2 * (ub - lb)
        stagnation = 0
        max_stagnation = max(5, int(0.1 * budget))
        while evals < budget:
            # Generate candidate by perturbing each dimension with its step size
            perturbation = rng.normal(0, 1, dim) * step_sizes
            x_cand = best_x + perturbation
            x_cand = np.clip(x_cand, lb, ub)
            val_cand = func(x_cand)
            evals += 1
            if evals > budget:
                break
            if val_cand < best_val:
                best_val = val_cand
                best_x = x_cand.copy()
                report_best(best_val, best_x)
                step_sizes *= 1.2
                stagnation = 0
            else:
                stagnation += 1
                step_sizes *= 0.95
                if np.any(step_sizes < 1e-10):
                    step_sizes = 0.01 * (ub - lb)
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
                        step_sizes = 0.2 * (ub - lb)
                        stagnation = 0
                    else:
                        break
            # Keep step sizes within bounds
            step_sizes = np.clip(step_sizes, 1e-10, 0.5 * (ub - lb))
        
        # Fallback if no evaluations (should not happen)
        if evals == 0:
            x = lb + (ub - lb) * rng.rand(dim)
            best_val = func(x)
            best_x = x.copy()
            report_best(best_val, best_x)
        
        return (best_val, best_x)