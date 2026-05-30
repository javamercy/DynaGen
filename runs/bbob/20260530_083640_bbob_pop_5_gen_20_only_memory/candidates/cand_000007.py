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

        # initial point
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)

        # step sizes per dimension
        step = 0.2 * (ub - lb)
        min_step = 1e-10 * (ub - lb)

        stagnation_limit = max(2 * dim, int(0.05 * budget))
        no_improve = 0

        while evals < budget:
            improved = False
            # coordinate polling
            for i in range(dim):
                if evals >= budget:
                    break
                # positive direction
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    improved = True
                    no_improve = 0
                    break
                # negative direction
                trial[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    improved = True
                    no_improve = 0
                    break
                else:
                    step[i] = max(step[i] * 0.5, min_step[i])

            if improved:
                continue

            # random direction if no coordinate improvement
            if evals < budget:
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                trial = np.clip(best_x + step * direction, lb, ub)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step = np.minimum(step * 2, ub - lb)
                    no_improve = 0
                else:
                    no_improve += 1
                    step = np.maximum(step * 0.5, min_step)

            # check stagnation for restart
            if no_improve >= stagnation_limit:
                # restart from new random point
                best_x = lb + rng.rand(dim) * (ub - lb)
                best_f = func(best_x)
                evals += 1
                report_best(best_f, best_x)
                step = 0.2 * (ub - lb)
                no_improve = 0

        return best_f, best_x