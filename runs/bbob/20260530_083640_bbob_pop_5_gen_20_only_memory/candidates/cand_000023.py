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
        # Initial point
        best_x = lb + self.rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)
        # Step sizes
        step = 0.2 * (ub - lb)
        # Stagnation parameters
        max_failures = max(dim * 2, 20)
        failure_counter = 0
        # Main loop
        while evals < self.budget:
            success = False
            # Coordinate polling
            perm = self.rng.permutation(dim)
            for i in perm:
                if evals >= self.budget:
                    break
                # Positive direction
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    success = True
                    break
                # Negative direction
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    success = True
                    break
                else:
                    step[i] = max(step[i] * 0.5, (ub[i] - lb[i]) * 1e-10)
            # Random direction poll if no success
            if not success and evals < self.budget:
                direction = self.rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                # Use larger step for exploration
                trial = np.clip(best_x + 2 * step * direction, lb, ub)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step = np.minimum(step * 2, ub - lb)
                    success = True
            # Update failure counter
            if success:
                failure_counter = 0
            else:
                failure_counter += 1
            # Random exploration with small probability (independent of failure)
            if evals < self.budget and self.rng.rand() < 0.05:
                trial = lb + self.rng.rand(dim) * (ub - lb)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step = 0.2 * (ub - lb)
                    failure_counter = 0
            # Restart if stagnation
            if failure_counter >= max_failures and evals < self.budget:
                trial = lb + self.rng.rand(dim) * (ub - lb)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                step = 0.2 * (ub - lb)
                failure_counter = 0
        return best_f, best_x