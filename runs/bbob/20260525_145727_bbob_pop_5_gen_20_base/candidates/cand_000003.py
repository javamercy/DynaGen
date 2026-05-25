import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Random initial point
        best_x = np.random.uniform(lb, ub, size=dim)
        best_val = func(best_x)
        report_best(best_val, best_x)
        remaining = budget - 1

        if remaining <= 0:
            return best_val, best_x

        # Adaptive radius (as fraction of bounds)
        radius = 0.2 * (ub - lb)  # initial radius
        max_stagnation = max(10, dim)
        stagnation = 0

        while remaining > 0:
            # Random direction
            direction = np.random.normal(0, 1, dim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = np.random.uniform(-1, 1, dim)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    direction = np.ones(dim)
                    norm = np.sqrt(dim)
            direction = direction / norm

            candidate = best_x + direction * radius
            candidate = np.clip(candidate, lb, ub)
            new_val = func(candidate)
            remaining -= 1

            if new_val < best_val:
                best_val = new_val
                best_x = candidate
                report_best(best_val, best_x)
                radius *= 1.2  # expand
                stagnation = 0
            else:
                radius *= 0.8  # contract
                stagnation += 1

            if stagnation >= max_stagnation and remaining > 0:
                # Restart: new random point
                new_x = np.random.uniform(lb, ub, size=dim)
                new_val = func(new_x)
                remaining -= 1
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x
                    report_best(best_val, best_x)
                radius = 0.2 * (ub - lb)
                stagnation = 0

        return best_val, best_x