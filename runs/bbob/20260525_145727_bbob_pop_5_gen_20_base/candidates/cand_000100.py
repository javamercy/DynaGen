import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        if budget == 0:
            return np.inf, np.zeros(dim)

        # initial random point
        x = lb + rng.rand(dim) * (ub - lb)
        best_val = func(x)
        best_x = x.copy()
        evals = 1
        report_best(best_val, best_x)

        if budget == 1:
            return best_val, best_x

        # adaptive step size (per dimension)
        step = 0.1 * (ub - lb)
        min_step = 1e-6 * (ub - lb)
        max_step = 0.2 * (ub - lb)

        while evals < budget:
            # order of coordinates shuffled
            perm = rng.permutation(dim)
            improved = False

            # pattern move
            for j in perm:
                if evals >= budget:
                    break
                # positive step
                x_new = x.copy()
                x_new[j] = x[j] + step[j]
                x_new[j] = np.clip(x_new[j], lb[j], ub[j])
                val = func(x_new)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x_new.copy()
                    x = x_new.copy()
                    improved = True
                    report_best(best_val, best_x)
                    continue  # move to next dimension

                # negative step
                x_new = x.copy()
                x_new[j] = x[j] - step[j]
                x_new[j] = np.clip(x_new[j], lb[j], ub[j])
                val = func(x_new)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x_new.copy()
                    x = x_new.copy()
                    improved = True
                    report_best(best_val, best_x)

            # adaptive step size
            if improved:
                step = np.minimum(step * 1.2, max_step)
            else:
                step = np.maximum(step * 0.5, min_step)

                # random perturbation with Cauchy distribution (heavy-tailed)
                if evals < budget and rng.rand() < 0.2:
                    perturb = rng.standard_cauchy(dim) * 0.1 * (ub - lb)
                    x_new = best_x + perturb
                    x_new = np.clip(x_new, lb, ub)
                    val = func(x_new)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_new.copy()
                        x = x_new.copy()
                        improved = True
                        report_best(best_val, best_x)

        return best_val, best_x