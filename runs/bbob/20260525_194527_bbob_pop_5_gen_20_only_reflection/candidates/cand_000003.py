import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        rang = ub - lb

        best_x = self.rng.uniform(lb, ub, size=self.dim)
        best_val = func(best_x)
        report_best(best_val, best_x)
        evals = 1

        radius = 0.1 * rang
        stagnation_count = 0
        stagnation_limit = max(10, self.budget // 50)

        while evals < self.budget:
            # decide action
            if stagnation_count >= stagnation_limit:
                # restart
                new_x = self.rng.uniform(lb, ub, size=self.dim)
                new_val = func(new_x)
                evals += 1
                if new_val < best_val:
                    best_val, best_x = new_val, new_x
                    report_best(best_val, best_x)
                    stagnation_count = 0
                else:
                    stagnation_count = 0
                radius = 0.1 * rang
                continue

            # local perturbation
            # sample direction uniformly on sphere
            direction = self.rng.normal(size=self.dim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = self.rng.uniform(-1, 1, size=self.dim)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            step = self.rng.uniform(0, radius)
            new_x = best_x + step * direction
            new_x = np.clip(new_x, lb, ub)
            new_val = func(new_x)
            evals += 1

            if new_val < best_val:
                best_val, best_x = new_val, new_x
                report_best(best_val, best_x)
                radius *= 1.1  # expand
                stagnation_count = 0
            else:
                radius *= 0.9  # shrink
                stagnation_count += 1

            # keep radius within sensible bounds
            radius = np.clip(radius, 0.01 * rang, 0.5 * rang)

        return best_val, best_x