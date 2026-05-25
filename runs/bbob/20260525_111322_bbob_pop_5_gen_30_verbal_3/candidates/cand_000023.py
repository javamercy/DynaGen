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
        best_x = None
        best_f = np.inf
        evals = 0

        # Latin hypercube initialization
        n_init = min(self.budget, max(2 * self.dim, 1))
        points = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            perm = self.rng.permutation(n_init)
            u = self.rng.rand(n_init)
            points[:, i] = (perm + u) / n_init
        points = lb + points * (ub - lb)
        for i in range(n_init):
            if evals >= self.budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Pattern search with restarts
        step_initial = 0.1 * np.mean(ub - lb)
        while evals < self.budget:
            step = step_initial
            # Generate directions: coordinate axes and random unit vectors
            directions = []
            for i in range(self.dim):
                e = np.zeros(self.dim)
                e[i] = 1.0
                directions.append(e)
                directions.append(-e)
            # Add random directions
            for _ in range(self.dim):
                d = self.rng.randn(self.dim)
                d /= np.linalg.norm(d) + 1e-15
                directions.append(d)
                directions.append(-d)
            # Shuffle directions for randomness
            self.rng.shuffle(directions)

            improved = True
            while improved and evals < self.budget:
                improved = False
                for d in directions:
                    if evals >= self.budget:
                        break
                    candidate = best_x + step * d
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
                if step < 1e-12:
                    break
            # Restart if step too small and enough budget left
            if step < 1e-12 and self.budget - evals >= 2 * self.dim:
                n_restart = min(self.budget - evals, max(2 * self.dim, 1))
                restart_points = np.zeros((n_restart, self.dim))
                for i in range(self.dim):
                    perm = self.rng.permutation(n_restart)
                    u = self.rng.rand(n_restart)
                    restart_points[:, i] = (perm + u) / n_restart
                restart_points = lb + restart_points * (ub - lb)
                for i in range(n_restart):
                    if evals >= self.budget:
                        break
                    x = restart_points[i]
                    f = func(x)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                step_initial = 0.1 * np.mean(ub - lb)
            else:
                break
        return best_f, best_x