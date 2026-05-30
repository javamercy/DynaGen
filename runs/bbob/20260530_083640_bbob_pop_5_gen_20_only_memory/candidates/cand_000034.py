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
        budget = self.budget
        rng = self.rng

        best_x = lb + rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)

        step = 0.2 * (ub - lb)
        max_stagnation = max(1, int(dim * 2))

        while evals < budget:
            success = False
            perm = rng.permutation(dim)
            for i in perm:
                if evals >= budget:
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

            if not success and evals < budget:
                # Cauchy perturbation
                direction = rng.standard_cauchy(dim)
                trial = np.clip(best_x + step * direction, lb, ub)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step = np.minimum(step * 2, ub - lb)
                    stagnation = 0
                else:
                    stagnation += 1
                    if stagnation >= max_stagnation:
                        # Restart
                        best_x = lb + rng.rand(dim) * (ub - lb)
                        f = func(best_x)
                        evals += 1
                        if f < best_f:
                            best_f = f
                            report_best(best_f, best_x.copy())
                        step = 0.2 * (ub - lb)
                        stagnation = 0
            else:
                stagnation = 0

        return best_f, best_x