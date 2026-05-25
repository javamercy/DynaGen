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

        n_init = min(budget, max(2 * dim, 1))
        # Latin hypercube sampling
        points = np.empty((n_init, dim))
        for i in range(dim):
            perm = rng.permutation(n_init)
            points[:, i] = lb[i] + (perm + 0.5) / n_init * (ub[i] - lb[i])
        
        best_f = np.inf
        best_x = np.zeros(dim)
        evals = 0
        for i in range(n_init):
            if evals >= budget:
                break
            f = func(points[i])
            evals += 1
            if f < best_f:
                best_f = f
                best_x = points[i].copy()
                report_best(best_f, best_x)

        # Pattern search
        step = 0.1 * np.mean(ub - lb)
        initial_step = step
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)
        
        current_x = best_x.copy()
        current_f = best_f

        while evals < budget:
            improved = False
            for d in directions:
                if evals >= budget:
                    break
                candidate = current_x + step * d
                candidate = np.clip(candidate, lb, ub)
                f_val = func(candidate)
                evals += 1
                if f_val < current_f:
                    current_f = f_val
                    current_x = candidate.copy()
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                        report_best(best_f, best_x)
                    improved = True
                    break
            if improved:
                step *= 1.2
            else:
                step *= 0.5
                if step < 1e-15 and evals < budget:
                    # restart: reset step and start from a random point
                    step = initial_step
                    candidate = lb + rng.rand(dim) * (ub - lb)
                    f_val = func(candidate)
                    evals += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
                    current_f = f_val
                    current_x = candidate.copy()
                elif step < 1e-15:
                    break

        return best_f, best_x