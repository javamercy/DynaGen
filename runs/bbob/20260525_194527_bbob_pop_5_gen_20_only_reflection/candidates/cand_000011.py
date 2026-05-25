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
        stagnation_limit = 7

        while evals < self.budget:
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
                radius *= 1.2  # expand
                stagnation_count = 0
                # line search extension: try to continue in same direction (only 1 step)
                if evals < self.budget:
                    extra_step = radius * 1.5
                    candidate = best_x + extra_step * direction
                    candidate = np.clip(candidate, lb, ub)
                    candidate_val = func(candidate)
                    evals += 1
                    if candidate_val < best_val:
                        best_val, best_x = candidate_val, candidate
                        report_best(best_val, best_x)
                        radius = extra_step
            else:
                radius *= 0.7  # shrink
                stagnation_count += 1

            # keep radius within sensible bounds
            radius = np.clip(radius, 0.01 * rang, 0.5 * rang)

        return best_val, best_x