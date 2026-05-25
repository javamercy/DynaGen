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
        # Latin hypercube initialization
        n_init = min(budget, max(2 * dim, 5))
        points = np.zeros((n_init, dim))
        for i in range(dim):
            perm = rng.permutation(n_init)
            u = rng.rand(n_init)
            points[:, i] = (perm + u) / n_init
        points = lb + points * (ub - lb)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(n_init):
            if evals >= budget:
                break
            x = np.clip(points[i], lb, ub)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        # Pattern search with random directions
        step = 0.1 * np.mean(ub - lb)
        while evals < budget and step > 1e-15:
            improved = False
            # generate a random direction
            d = rng.randn(dim)
            d /= np.linalg.norm(d)
            # poll both directions
            for sign in [1.0, -1.0]:
                if evals >= budget:
                    break
                candidate = best_x + sign * step * d
                candidate = np.clip(candidate, lb, ub)
                f_val = func(candidate)
                evals += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = candidate.copy()
                    improved = True
                    report_best(best_f, best_x)
                    break
            if improved:
                step *= 1.2
            else:
                step *= 0.5
            # restart if step too small and budget left
            if step < 1e-15 and evals < budget:
                # generate new LHS points
                n_restart = min(budget - evals, max(2 * dim, 3))
                points = np.zeros((n_restart, dim))
                for i in range(dim):
                    perm = rng.permutation(n_restart)
                    u = rng.rand(n_restart)
                    points[:, i] = (perm + u) / n_restart
                points = lb + points * (ub - lb)
                for i in range(n_restart):
                    if evals >= budget:
                        break
                    x = np.clip(points[i], lb, ub)
                    f = func(x)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                step = 0.1 * np.mean(ub - lb)
        return best_f, best_x