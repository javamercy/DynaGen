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
        # LHS initialization
        n_init = min(self.budget, max(2 * self.dim, 1))
        best_x = None
        best_f = np.inf
        points = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            perm = self.rng.permutation(n_init)
            u = self.rng.rand(n_init)
            points[:, i] = (perm + u) / n_init
        points = lb + points * (ub - lb)
        evals = 0
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
        # History for covariance
        history = [best_x.copy()]
        step = 0.1 * np.mean(ub - lb)
        # Coordinate directions
        coord_dirs = []
        for i in range(self.dim):
            e = np.zeros(self.dim)
            e[i] = 1.0
            coord_dirs.append(e)
            coord_dirs.append(-e)
        eig_dirs = []
        # Main loop
        while evals < self.budget:
            # Recompute eigen directions periodically
            if len(history) >= self.dim and (evals % (10 * self.dim) == 0):
                arr = np.array(history)
                cov = np.cov(arr, rowvar=False)
                if cov.shape[0] == self.dim:
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    eig_dirs = []
                    for v in eigenvectors.T:
                        eig_dirs.append(v)
                        eig_dirs.append(-v)
            dirs = coord_dirs + eig_dirs
            improved = False
            for d in dirs:
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
                    history.append(best_x.copy())
                    if len(history) > 3 * self.dim:
                        history.pop(0)
                    break
            if improved:
                step *= 1.2
            else:
                step *= 0.5
                # Restart if step too small and budget remains
                if step < 1e-12 and evals < self.budget:
                    n_restart = min(self.budget - evals, max(2 * self.dim, 1))
                    new_points = np.zeros((n_restart, self.dim))
                    for i in range(self.dim):
                        perm = self.rng.permutation(n_restart)
                        u = self.rng.rand(n_restart)
                        new_points[:, i] = (perm + u) / n_restart
                    new_points = lb + new_points * (ub - lb)
                    best_local_f = best_f
                    best_local_x = best_x.copy()
                    for i in range(n_restart):
                        if evals >= self.budget:
                            break
                        x = new_points[i]
                        f = func(x)
                        evals += 1
                        if f < best_local_f:
                            best_local_f = f
                            best_local_x = x.copy()
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                                report_best(best_f, best_x)
                    best_x = best_local_x.copy()
                    step = 0.1 * np.mean(ub - lb)
                    history = [best_x.copy()]
                    eig_dirs = []
        return best_f, best_x