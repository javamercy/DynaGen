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
        rng = self.rng
        budget = self.budget

        # Latin hypercube initialization
        n_init = min(budget, max(2 * dim, 1))
        best_x = None
        best_f = np.inf
        evals = 0
        points = np.zeros((n_init, dim))
        for i in range(dim):
            perm = rng.permutation(n_init)
            u = rng.rand(n_init)
            points[:, i] = (perm + u) / n_init
        points = lb + points * (ub - lb)
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

        # Pattern search with covariance-guided random directions
        initial_step = 0.1 * np.mean(ub - lb)
        step = initial_step
        # Coordinate directions
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)
        # Covariance matrix for random directions
        C = np.eye(dim)
        decay = 0.1
        success_steps = []  # store recent successful step vectors

        while evals < budget:
            improved = False
            # Poll coordinate directions
            for d in directions:
                if evals >= budget:
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
                    # Record successful step
                    s = step * d  # actual step vector
                    success_steps.append(s)
                    if len(success_steps) > 2 * dim:
                        success_steps.pop(0)
                    # Update covariance
                    C = (1 - decay) * C + decay * np.outer(s, s)
                    break
            if improved:
                step *= 1.2
                continue
            # If no improvement from coordinates, try random directions from covariance
            n_rand = min(2, dim)
            if len(success_steps) > 0:
                # Ensure C is positive definite
                C_reg = C + 1e-12 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C_reg)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
            else:
                L = np.eye(dim)
            for _ in range(n_rand):
                if evals >= budget:
                    break
                z = rng.randn(dim)
                rand_dir = L @ z
                norm = np.linalg.norm(rand_dir)
                if norm > 0:
                    rand_dir = rand_dir / norm * step
                else:
                    rand_dir = np.zeros(dim)
                candidate = best_x + rand_dir
                candidate = np.clip(candidate, lb, ub)
                f_val = func(candidate)
                evals += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = candidate.copy()
                    improved = True
                    report_best(best_f, best_x)
                    s = rand_dir
                    success_steps.append(s)
                    if len(success_steps) > 2 * dim:
                        success_steps.pop(0)
                    C = (1 - decay) * C + decay * np.outer(s, s)
                    step *= 1.2
                    break
            if not improved:
                step *= 0.5
                # Restart if step too small
                if step < 1e-12 * initial_step:
                    remaining = budget - evals
                    if remaining > 0:
                        n_restart = min(remaining, max(2 * dim, 1))
                        # Generate new LHS points
                        restart_points = np.zeros((n_restart, dim))
                        for i in range(dim):
                            perm = rng.permutation(n_restart)
                            u = rng.rand(n_restart)
                            restart_points[:, i] = (perm + u) / n_restart
                        restart_points = lb + restart_points * (ub - lb)
                        for i in range(n_restart):
                            if evals >= budget:
                                break
                            x = restart_points[i]
                            f = func(x)
                            evals += 1
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                                report_best(best_f, best_x)
                        step = initial_step
                        C = np.eye(dim)
                        success_steps.clear()
        return best_f, best_x