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

        # Pattern search parameters
        initial_step = 0.1 * np.mean(ub - lb)
        step = initial_step
        # Coordinate directions
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)
        # Covariance matrix
        C = np.eye(dim)
        decay = 0.1
        # Evolution path
        c_c = 0.1
        c_cov = 0.1
        p_c = np.zeros(dim)
        # Track consecutive failures for local refinement
        consecutive_failures = 0

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
                    s = step * d
                    # Update evolution path
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * s / step
                    # Update covariance with rank-one from evolution path
                    C = (1 - c_cov) * C + c_cov * np.outer(p_c, p_c)
                    # Also rank-one from step
                    C = (1 - decay) * C + decay * np.outer(s, s)
                    # Ensure positive definiteness
                    C += 1e-12 * np.eye(dim)
                    step *= 1.2
                    consecutive_failures = 0
                    break
            if improved:
                continue
            # If no improvement from coordinates, try random directions from covariance
            n_rand = min(2, dim)
            C_reg = C + 1e-12 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C_reg)
            except np.linalg.LinAlgError:
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
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * s / step
                    C = (1 - c_cov) * C + c_cov * np.outer(p_c, p_c)
                    C = (1 - decay) * C + decay * np.outer(s, s)
                    C += 1e-12 * np.eye(dim)
                    step *= 1.2
                    consecutive_failures = 0
                    break
            if improved:
                continue
            # No improvement
            step *= 0.5
            consecutive_failures += 1
            # Local refinement if stuck
            if consecutive_failures >= 2 * dim:
                n_local = min(budget - evals, 2 * dim)
                local_step = step / 2
                for _ in range(n_local):
                    if evals >= budget:
                        break
                    offset = rng.randn(dim) * local_step
                    candidate = best_x + offset
                    candidate = np.clip(candidate, lb, ub)
                    f_val = func(candidate)
                    evals += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
                        step *= 1.2
                        consecutive_failures = 0
                        break
            # Restart if step too small
            if step < 1e-12 * initial_step:
                remaining = budget - evals
                if remaining > 0:
                    n_restart = min(remaining, max(2 * dim, 1))
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
                    p_c = np.zeros(dim)
                    consecutive_failures = 0
        return best_f, best_x