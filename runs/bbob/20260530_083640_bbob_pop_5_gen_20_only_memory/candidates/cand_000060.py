import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)
        
        best_val = np.inf
        best_x = None
        evals = 0
        
        # Initial Latin Hypercube sampling
        n_init = min(max(4, 2 * dim), budget // 4)
        if n_init < 1:
            n_init = 1
        lhs = np.zeros((n_init, dim))
        for j in range(dim):
            perm = rng.permutation(n_init)
            for i in range(n_init):
                lhs[i, j] = (perm[i] + rng.uniform(0, 1)) / n_init
        init_points = lb + (ub - lb) * lhs
        for i in range(n_init):
            x = init_points[i]
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x
        
        # Main loop with adaptive step size
        remaining = budget - evals
        step0 = 0.5 * (ub - lb)  # initial step size
        for i in range(remaining):
            if evals >= budget:
                break
            # Step size decays quadratically
            # t = i / remaining  # fraction of remaining used
            # step = step0 * (1 - t)**1.5  # decay faster
            # Alternative: linear decay to 0.01
            t = i / max(remaining, 1)
            step = (0.5 * (1 - t) + 0.01 * t) * (ub - lb)
            # Generate candidate
            trial = best_x + rng.randn(dim) * step
            trial = np.clip(trial, lb, ub)
            val = func(trial)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                report_best(best_val, best_x)
        
        return best_val, best_x