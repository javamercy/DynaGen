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
        # Adaptive initial step size: 0.2 * range / sqrt(dim)
        self.initial_step = 0.2 * (ub - lb) / np.sqrt(dim)
        step = self.initial_step.copy()
        # Initial perturbation probability (will be scheduled)
        p0 = 0.2
        while evals < self.budget:
            success = False
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
            # Restart with decaying probability
            if not success and evals < self.budget:
                p = p0 * (1.0 - evals / self.budget)
                if self.rng.rand() < p:
                    trial = lb + self.rng.rand(dim) * (ub - lb)
                    f = func(trial)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = trial
                        report_best(best_f, best_x)
                    step = self.initial_step.copy()
        return best_f, best_x