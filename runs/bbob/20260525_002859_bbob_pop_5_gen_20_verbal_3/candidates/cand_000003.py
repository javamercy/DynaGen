import numpy as np
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.best_value = None
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Initial point
        x0 = lb + rng.rand(dim) * (ub - lb)
        f0 = func(x0)
        self.best_value = f0
        self.best_x = x0.copy()
        best_value = f0
        best_x = x0.copy()
        # Call report_best initially (should be fine)
        # report_best(best_value, best_x)  # uncomment if environment provides it

        # Adaptive radius
        radius = (ub - lb).mean() / 5.0
        # Stagnation counter
        stagnation = 0
        max_stagnation = max(10, 2 * dim)
        evaluations = 1

        while evaluations < budget:
            # Check if restart is needed
            if stagnation >= max_stagnation:
                # Restart: sample new random point
                x = lb + rng.rand(dim) * (ub - lb)
                radius = (ub - lb).mean() / 5.0
                stagnation = 0
            else:
                # Generate candidate around best_x
                x = best_x + radius * rng.randn(dim)
                x = np.clip(x, lb, ub)

            if evaluations >= budget:
                break
            f = func(x)
            evaluations += 1

            if f < best_value:
                best_value = f
                best_x = x.copy()
                radius *= 1.2  # increase radius
                stagnation = 0
                # report_best(best_value, best_x)
            else:
                radius *= 0.8  # decrease radius
                stagnation += 1

            # Ensure radius doesn't become too small
            radius = max(radius, 1e-10 * (ub - lb).mean())

        return best_value, best_x