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
        width = ub - lb
        mean_width = np.mean(width)

        # initial random point
        best_x = self.rng.uniform(lb, ub, size=self.dim)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # adaptive parameters
        radius = 0.2 * mean_width
        max_failures = max(5, 2 * self.dim)
        failures = 0

        while evals < self.budget:
            if failures >= max_failures:
                # restart
                x = self.rng.uniform(lb, ub, size=self.dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x
                    report_best(best_val, best_x)
                failures = 0
                radius = 0.2 * mean_width
            else:
                # sample a random direction
                direction = self.rng.normal(0, 1, size=self.dim)
                step = direction * radius * width

                # try positive step
                x_plus = best_x + step
                x_plus = np.clip(x_plus, lb, ub)
                val_plus = func(x_plus)
                evals += 1
                if val_plus < best_val:
                    best_val = val_plus
                    best_x = x_plus
                    report_best(best_val, best_x)
                    radius *= 1.1
                    failures = 0
                else:
                    # try negative step
                    x_minus = best_x - step
                    x_minus = np.clip(x_minus, lb, ub)
                    val_minus = func(x_minus)
                    evals += 1
                    if val_minus < best_val:
                        best_val = val_minus
                        best_x = x_minus
                        report_best(best_val, best_x)
                        radius *= 1.1
                        failures = 0
                    else:
                        failures += 1
                        radius *= 0.9

                # ensure radius does not become too small
                if radius < 1e-8 * mean_width:
                    radius = 1e-8 * mean_width

                # break if budget exhausted
                if evals >= self.budget:
                    break

        return best_val, best_x