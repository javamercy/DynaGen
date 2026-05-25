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
        rng = self.rng
        budget = self.budget

        # Initial point
        best_x = rng.uniform(lb, ub)
        best_f = func(best_x)
        report_best(best_f, best_x)
        evals = 1

        # Step sizes as fraction of range
        step = 0.2 * (ub - lb)
        min_step = 1e-3 * (ub - lb)
        init_step = step.copy()

        # Stagnation detection
        consecutive_failures = 0
        stagnation_limit = max(10, dim * 10)

        while evals < budget:
            # Random coordinate
            coord = rng.randint(dim)
            # Random direction
            direction = 1 if rng.rand() < 0.5 else -1

            # Try positive direction
            candidate = best_x.copy()
            candidate[coord] += direction * step[coord]
            candidate = np.clip(candidate, lb, ub)
            f = func(candidate)
            evals += 1

            if f < best_f:
                best_f = f
                best_x = candidate
                report_best(best_f, best_x)
                step[coord] *= 1.5
                consecutive_failures = 0
                continue
            else:
                consecutive_failures += 1

            # Try opposite direction if budget left
            if evals < budget:
                candidate2 = best_x.copy()
                candidate2[coord] -= direction * step[coord]
                candidate2 = np.clip(candidate2, lb, ub)
                f2 = func(candidate2)
                evals += 1
                if f2 < best_f:
                    best_f = f2
                    best_x = candidate2
                    report_best(best_f, best_x)
                    step[coord] *= 1.2
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    step[coord] *= 0.5
            else:
                step[coord] *= 0.5

            # Ensure minimum step
            step[coord] = max(step[coord], min_step[coord])

            # Restart if stagnation
            if consecutive_failures >= stagnation_limit:
                step[:] = init_step
                consecutive_failures = 0

        return best_f, best_x