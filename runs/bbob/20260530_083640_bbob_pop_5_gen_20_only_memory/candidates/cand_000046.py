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
        stagnation = 0
        max_stag = max(1, budget // 8)
        restart_interval = max(1, budget // 4)
        last_restart_evals = 0
        step_increase = 1.5
        step_decrease = 0.75
        perturbation_prob = 0.25

        while evals < budget:
            # Scheduled restart
            if evals - last_restart_evals >= restart_interval and evals < budget:
                new_x = lb + rng.rand(dim) * (ub - lb)
                new_f = func(new_x)
                evals += 1
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                    report_best(best_f, best_x)
                step = 0.2 * (ub - lb)
                stagnation = 0
                last_restart_evals = evals
                continue

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
                    step[i] = min(step[i] * step_increase, ub[i] - lb[i])
                    success = True
                    stagnation = 0
                    break
                # Negative direction
                trial[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step[i] = min(step[i] * step_increase, ub[i] - lb[i])
                    success = True
                    stagnation = 0
                    break
                else:
                    step[i] = max(step[i] * step_decrease, (ub[i] - lb[i]) * 1e-10)

            if not success and evals < budget:
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
                    step = np.minimum(step * step_increase, ub - lb)
                    success = True
                    stagnation = 0
                else:
                    stagnation += 1

            # Random perturbation
            if evals < budget and rng.uniform() < perturbation_prob:
                scale = rng.uniform(0.1, 0.5)
                perturbation = scale * (ub - lb) * rng.randn(dim)
                trial = np.clip(best_x + perturbation, lb, ub)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step = np.minimum(step * step_increase, ub - lb)
                    success = True
                    stagnation = 0

            # Stagnation restart
            if stagnation >= max_stag and evals < budget:
                new_x = lb + rng.rand(dim) * (ub - lb)
                new_f = func(new_x)
                evals += 1
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                    report_best(best_f, best_x)
                step = 0.2 * (ub - lb)
                stagnation = 0
                last_restart_evals = evals

        return best_f, best_x