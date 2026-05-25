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
        range_len = ub - lb
        # Latin hypercube initialization
        n_init = min(budget, max(2 * dim, 1))
        points = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            points[:, d] = lb[d] + (perm + rng.uniform(size=n_init)) / n_init * (ub[d] - lb[d])
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(n_init):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        # Initial step and covariance
        step = 0.1 * np.mean(range_len)
        C = np.eye(dim)
        lr = 0.1
        # Pre-generate coordinate directions
        coord_dirs = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            coord_dirs.append(d)
            coord_dirs.append(-d)
        while evals < budget:
            improved = False
            # Generate random directions from current covariance
            num_random = min(2 * dim, budget - evals)  # limit to avoid overshoot
            if num_random > 0:
                try:
                    # Ensure positive definite
                    C_reg = C + 1e-15 * np.eye(dim)
                    samples = rng.multivariate_normal(np.zeros(dim), C_reg, size=num_random)
                    # Normalize to unit length
                    norms = np.linalg.norm(samples, axis=1, keepdims=True) + 1e-15
                    random_dirs = samples / norms
                except:
                    random_dirs = np.zeros((0, dim))
            else:
                random_dirs = np.zeros((0, dim))
            # Combine directions: coordinate first then random
            directions = coord_dirs + [random_dirs[i] for i in range(len(random_dirs))]
            for d in directions:
                if evals >= budget:
                    break
                candidate = best_x + step * d
                candidate = np.clip(candidate, lb, ub)
                f = func(candidate)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    improved = True
                    # Update covariance with successful direction (normalized d)
                    d_norm = np.linalg.norm(d) + 1e-15
                    d_unit = d / d_norm
                    C = (1 - lr) * C + lr * np.outer(d_unit, d_unit)
                    C += 1e-15 * np.eye(dim)  # avoid singular
                    break
            if improved:
                step *= 1.2
            else:
                step *= 0.5
                if step < 1e-12 and evals < budget:
                    # Restart: LHS on remaining budget
                    remaining = budget - evals
                    if remaining > 0:
                        n_restart = min(remaining, max(2 * dim, 1))
                        points_restart = np.zeros((n_restart, dim))
                        for d in range(dim):
                            perm = rng.permutation(n_restart)
                            points_restart[:, d] = lb[d] + (perm + rng.uniform(size=n_restart)) / n_restart * (ub[d] - lb[d])
                        for i in range(n_restart):
                            if evals >= budget:
                                break
                            x = points_restart[i]
                            f = func(x)
                            evals += 1
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                                report_best(best_f, best_x)
                        # Reset step and covariance
                        step = 0.1 * np.mean(range_len)
                        C = np.eye(dim)
        return best_f, best_x