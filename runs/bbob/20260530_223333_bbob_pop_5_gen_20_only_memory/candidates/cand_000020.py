import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)

        best_val = np.inf
        best_x = None

        def evaluate(x):
            nonlocal best_val, best_x
            val = func(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            return val

        # Phase 1: Pattern search from random start
        x = rng.uniform(lb, ub)
        f_x = evaluate(x)
        evals_used = 1
        local_best_val = best_val
        local_best_x = x.copy()
        step = 0.2 * (ub - lb)
        min_step = 1e-8 * (ub - lb)
        no_improve = 0
        max_no_improve = max(2 * dim, 15)

        while evals_used < budget * 0.7 and no_improve < max_no_improve and np.any(step > min_step):
            improved = False
            for d in rng.permutation(dim):
                if evals_used >= budget * 0.7:
                    break
                x_new = x.copy()
                x_new[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val = evaluate(x_new)
                    evals_used += 1
                    if val < local_best_val:
                        local_best_val = val
                        local_best_x = x_new.copy()
                        x = x_new.copy()
                        improved = True
                        continue
                x_new = x.copy()
                x_new[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val = evaluate(x_new)
                    evals_used += 1
                    if val < local_best_val:
                        local_best_val = val
                        local_best_x = x_new.copy()
                        x = x_new.copy()
                        improved = True

            if improved:
                direction = x - local_best_x
                x_new = x + direction
                x_new = np.clip(x_new, lb, ub)
                if np.any(x_new != x) and evals_used < budget * 0.7:
                    val = evaluate(x_new)
                    evals_used += 1
                    if val < local_best_val:
                        local_best_val = val
                        local_best_x = x_new.copy()
                        x = x_new.copy()
                step *= 1.2
                no_improve = 0
            else:
                step *= 0.5
                x = local_best_x.copy()
                no_improve += 1

        # Phase 2: Local intensification around best
        remaining = budget - evals_used
        if remaining > 0:
            x = best_x.copy()
            radius = 0.1 * (ub - lb)
            min_radius = 1e-6 * (ub - lb)
            for _ in range(remaining):
                if evals_used >= budget:
                    break
                candidate = x + radius * rng.standard_cauchy(dim)
                candidate = np.clip(candidate, lb, ub)
                val = evaluate(candidate)
                evals_used += 1
                if val < best_val:
                    x = candidate.copy()
                    radius = radius * 1.1
                else:
                    radius = radius * 0.9
                radius = np.maximum(radius, min_radius)

        return best_val, best_x