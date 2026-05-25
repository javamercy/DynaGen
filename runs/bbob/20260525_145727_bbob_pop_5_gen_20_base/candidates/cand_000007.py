import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial random point
        best_x = np.random.uniform(lb, ub, size=dim)
        best_val = func(best_x)
        report_best(best_val, best_x)
        remaining = budget - 1

        if remaining <= 0:
            return best_val, best_x

        # Adaptive step size (as fraction of bounds)
        step_size = 0.1 * (ub - lb)  # smaller initial step for exploitation
        max_stagnation = max(15, 2 * dim)
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

            # Main candidate
            candidate = best_x + direction * step_size
            candidate = np.clip(candidate, lb, ub)
            new_val = func(candidate)
            remaining -= 1

            if new_val < best_val:
                best_val = new_val
                best_x = candidate
                report_best(best_val, best_x)
                step_size *= 1.5  # expand
                stagnation = 0

                # Local search along the same direction (up to 3 steps)
                for _ in range(3):
                    if remaining <= 0:
                        break
                    local_candidate = best_x + direction * step_size
                    local_candidate = np.clip(local_candidate, lb, ub)
                    local_val = func(local_candidate)
                    remaining -= 1
                    if local_val < best_val:
                        best_val = local_val
                        best_x = local_candidate
                        report_best(best_val, best_x)
                        step_size *= 1.5
                    else:
                        break
            else:
                step_size *= 0.6  # contract
                stagnation += 1

            if stagnation >= max_stagnation and remaining > 0:
                # Restart
                new_x = np.random.uniform(lb, ub, size=dim)
                new_val = func(new_x)
                remaining -= 1
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x
                    report_best(best_val, best_x)
                step_size = 0.1 * (ub - lb)
                stagnation = 0

        return best_val, best_x