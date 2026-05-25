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
        
        # Parameters for self-adaptation
        tau = 1.0 / np.sqrt(2 * dim)
        tau_prime = 1.0 / np.sqrt(2 * np.sqrt(dim))
        
        # Initial exploration (10% of budget, at least 1)
        n_global = max(1, int(0.1 * budget))
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
        
        # Initialize step sizes (per dimension)
        sigma = 0.2 * (ub - lb)
        
        # Self-adaptive (1+1)-ES
        stagnation = 0
        max_stagnation = max(10, int(0.1 * budget))  # restart after this many failures
        while evals < budget:
            # Mutate step sizes
            sigma_cand = sigma * np.exp(tau_prime * rng.normal() + tau * rng.normal(size=dim))
            # Clip step sizes to avoid extreme values
            sigma_cand = np.clip(sigma_cand, 1e-12 * (ub - lb), 0.5 * (ub - lb))
            # Mutation of position
            x_cand = best_x + sigma_cand * rng.normal(size=dim)
            x_cand = np.clip(x_cand, lb, ub)
            val_cand = func(x_cand)
            evals += 1
            if val_cand < best_val:
                best_val = val_cand
                best_x = x_cand.copy()
                sigma = sigma_cand.copy()
                report_best(best_val, best_x)
                stagnation = 0
            else:
                stagnation += 1
            # Restart if stagnation too high
            if stagnation >= max_stagnation and evals < budget:
                x_new = lb + (ub - lb) * rng.rand(dim)
                val_new = func(x_new)
                evals += 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
                sigma = 0.2 * (ub - lb)
                stagnation = 0
        
        # Fallback if no evaluations
        if evals == 0:
            x = lb + (ub - lb) * rng.rand(dim)
            best_val = func(x)
            best_x = x.copy()
            report_best(best_val, best_x)
        
        return (best_val, best_x)