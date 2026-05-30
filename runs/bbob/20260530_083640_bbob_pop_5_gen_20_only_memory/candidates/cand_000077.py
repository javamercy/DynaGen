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
        max_step = (ub - lb)

        stagnation_limit = max(dim, int(0.03 * budget))
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
                    step[i] = min(step[i] * 2, max_step[i])
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
                    step[i] = min(step[i] * 2, max_step[i])
                    improved = True
                    no_improve = 0
                    break
                else:
                    step[i] = max(step[i] * 0.5, min_step[i])

            if improved:
                continue

            # try random directions (two attempts)
            for _ in range(2):
                if evals >= budget:
                    break
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                # standard step
                trial = np.clip(best_x + step * direction, lb, ub)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step = np.minimum(step * 2, max_step)
                    improved = True
                    no_improve = 0
                    break
                else:
                    # long jump
                    trial = np.clip(best_x + 2.0 * step * direction, lb, ub)
                    f = func(trial)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = trial
                        report_best(best_f, best_x)
                        step = np.minimum(step * 2, max_step)
                        improved = True
                        no_improve = 0
                        break
            if improved:
                continue

            # try a perturbed version of current best (large mutation) if budget remains
            if evals < budget:
                # random perturbation with step size 0.5 * domain range
                perturb = 0.5 * (ub - lb) * rng.randn(dim)
                trial = np.clip(best_x + perturb, lb, ub)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step = 0.2 * (ub - lb)  # reset step
                    improved = True
                    no_improve = 0
                else:
                    no_improve += 1
                    step = np.maximum(step * 0.5, min_step)

            # check stagnation for restart
            if no_improve >= stagnation_limit:
                # restart from new random point
                new_x = lb + rng.rand(dim) * (ub - lb)
                f_new = func(new_x)
                evals += 1
                # also try a perturbed version of current best
                if evals < budget:
                    perturb = 0.5 * (ub - lb) * rng.randn(dim)
                    perturbed_x = np.clip(best_x + perturb, lb, ub)
                    f_pert = func(perturbed_x)
                    evals += 1
                    if f_pert < f_new:
                        new_x = perturbed_x
                        f_new = f_pert
                if f_new < best_f:
                    best_f = f_new
                    best_x = new_x
                    report_best(best_f, best_x)
                # reset state
                step = 0.2 * (ub - lb)
                no_improve = 0

        return best_f, best_x