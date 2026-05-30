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

        # Initial point
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)

        # Step sizes per dimension
        step = 0.2 * (ub - lb)
        # Success streak (per dimension) for adaptive step
        success_streak = np.zeros(dim, dtype=int)
        fail_streak = np.zeros(dim, dtype=int)

        x = best_x.copy()
        f = best_f
        stagnation = 0
        max_stag = max(1, budget // 10)

        while evals < budget:
            improved = False
            perm = rng.permutation(dim)
            for i in perm:
                if evals >= budget:
                    break
                # Positive direction
                trial = x.copy()
                trial[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                f_trial = func(trial)
                evals += 1
                if f_trial < f:
                    x = trial
                    f = f_trial
                    success_streak[i] += 1
                    fail_streak[i] = 0
                    if success_streak[i] >= 2:
                        step[i] = min(step[i] * 2, ub[i] - lb[i])
                        success_streak[i] = 0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                    improved = True
                    stagnation = 0
                    break
                # Negative direction
                trial[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                f_trial = func(trial)
                evals += 1
                if f_trial < f:
                    x = trial
                    f = f_trial
                    success_streak[i] += 1
                    fail_streak[i] = 0
                    if success_streak[i] >= 2:
                        step[i] = min(step[i] * 2, ub[i] - lb[i])
                        success_streak[i] = 0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                    improved = True
                    stagnation = 0
                    break
                else:
                    fail_streak[i] += 1
                    success_streak[i] = 0
                    if fail_streak[i] >= 2:
                        step[i] = max(step[i] * 0.5, (ub[i] - lb[i]) * 1e-10)
                        fail_streak[i] = 0

            if not improved and evals < budget:
                # Random direction poll
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                scale = np.mean(step)
                trial = np.clip(x + scale * direction, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < f:
                    x = trial
                    f = f_trial
                    step = np.minimum(step * 2, ub - lb)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                    stagnation = 0
                else:
                    stagnation += 1

            if stagnation >= max_stag and evals < budget:
                # Restart from best point with large perturbation
                x = best_x + 0.2 * (ub - lb) * rng.randn(dim)
                x = np.clip(x, lb, ub)
                f = func(x)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
                step = 0.2 * (ub - lb)
                success_streak = np.zeros(dim, dtype=int)
                fail_streak = np.zeros(dim, dtype=int)
                stagnation = 0

        return best_f, best_x