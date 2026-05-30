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

        # Initial point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # SA parameters
        T = 1.0
        cooling_rate = 0.95
        step = 0.2 * (ub - lb)
        min_step = 1e-8 * (ub - lb)
        max_step = 1.0 * (ub - lb)

        current_x = best_x.copy()
        current_val = best_val

        stagnation = 0
        max_stag = max(100, 20 * dim)

        while evals < budget:
            # Generate neighbor
            perturbation = rng.randn(dim) * step
            neighbor_x = np.clip(current_x + perturbation, lb, ub)
            neighbor_val = func(neighbor_x)
            evals += 1

            # Acceptance criterion
            delta = neighbor_val - current_val
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                current_x = neighbor_x
                current_val = neighbor_val
                if neighbor_val < best_val:
                    best_val = neighbor_val
                    best_x = neighbor_x.copy()
                    report_best(best_val, best_x)
                    stagnation = 0
                else:
                    stagnation += 1
                step = np.minimum(step * 1.1, max_step)
            else:
                stagnation += 1
                step = np.maximum(step * 0.9, min_step)

            # Update temperature
            T *= cooling_rate

            # Restart if stagnation
            if stagnation >= max_stag and evals < budget:
                new_x = rng.uniform(lb, ub)
                new_val = func(new_x)
                evals += 1
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
                current_x = new_x
                current_val = new_val
                step = 0.2 * (ub - lb)
                T = 1.0
                stagnation = 0

        return best_val, best_x