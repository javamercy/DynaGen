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

        best_x = np.random.uniform(lb, ub, size=dim)
        best_val = func(best_x)
        report_best(best_val, best_x)
        remaining = budget - 1

        if remaining <= 0:
            return best_val, best_x

        step_size = 0.1 * (ub - lb)
        max_stagnation = max(15, 2 * dim)
        stagnation = 0

        while remaining > 0:
            perturbation = np.random.uniform(-1, 1, dim) * step_size
            candidate = np.clip(best_x + perturbation, lb, ub)
            new_val = func(candidate)
            remaining -= 1

            if new_val < best_val:
                best_val = new_val
                best_x = candidate
                report_best(best_val, best_x)
                step_size *= 1.5
                stagnation = 0

                for _ in range(3):
                    if remaining <= 0:
                        break
                    local_candidate = np.clip(best_x + perturbation, lb, ub)
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
                step_size *= 0.6
                stagnation += 1

            if stagnation >= max_stagnation and remaining > 0:
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