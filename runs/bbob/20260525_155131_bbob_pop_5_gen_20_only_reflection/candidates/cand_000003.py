import numpy as np
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_x = None
        self.best_val = np.inf
        self.calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial point: uniform random within bounds
        x0 = self.rng.uniform(lb, ub, size=self.dim)
        val0 = func(x0)
        self.calls += 1
        self.best_x = x0.copy()
        self.best_val = val0
        report_best(self.best_val, self.best_x)

        # Parameters
        init_scale = 0.2 * (ub - lb)  # initial std dev for local search
        min_scale = 1e-3 * (ub - lb)
        scale = init_scale.copy()
        stagnation_limit = max(5, int(0.05 * self.budget))
        no_improve_count = 0
        restart_count = 0

        while self.calls < self.budget:
            # If many evaluations without improvement, restart
            if no_improve_count >= stagnation_limit:
                # Restart: new random point
                x = self.rng.uniform(lb, ub, size=self.dim)
                no_improve_count = 0
                scale = init_scale.copy()
                restart_count += 1
            else:
                # Local perturbation around best
                x = self.best_x + self.rng.normal(0, scale, size=self.dim)
                # Clip to bounds
                x = np.clip(x, lb, ub)
            
            val = func(x)
            self.calls += 1
            if val < self.best_val:
                self.best_val = val
                self.best_x = x.copy()
                report_best(self.best_val, self.best_x)
                no_improve_count = 0
                # Expand scale after improvement (optional)
                scale = np.minimum(scale * 1.2, init_scale)
            else:
                no_improve_count += 1
                # Shrink scale on no improvement
                scale = np.maximum(scale * 0.95, min_scale)

        return self.best_val, self.best_x