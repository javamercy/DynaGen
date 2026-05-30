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
        # Initial random point
        best_x = lb + self.rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)
        # Step sizes
        step = 0.2 * (ub - lb)
        # Main loop
        while evals < self.budget:
            success = False
            # Coordinate polling in random order
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
                    step[i] *= 1.5
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
                    step[i] *= 1.5
                    success = True
                    break
                else:
                    step[i] *= 0.5
            # If no success after full cycle, random perturbation with probability 0.1
            if not success and evals < self.budget and self.rng.rand() < 0.1:
                trial = lb + self.rng.rand(dim) * (ub - lb)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step = 0.2 * (ub - lb)  # reset steps
        return best_f, best_x