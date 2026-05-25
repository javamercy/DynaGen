import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        rng = self.rng
        budget = self.budget

        def lhs_points(n):
            points = np.zeros((n, dim))
            for i in range(dim):
                perm = rng.permutation(n)
                u = rng.rand(n)
                points[:, i] = (perm + u) / n
            return lb + points * (ub - lb)

        n_init = max(2 * dim, 1)
        n_init = min(n_init, budget)
        best_x = None
        best_f = np.inf
        evals = 0

        init_pts = lhs_points(n_init)
        for i in range(n_init):
            if evals >= budget:
                break
            x = init_pts[i]
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        if best_x is None:
            best_x = (lb + ub) / 2.0
            f = func(best_x)
            evals += 1
            best_f = f
            report_best(best_f, best_x)

        initial_step = 0.1 * np.mean(ub - lb)
        step = initial_step
        directions = [np.eye(dim)[i] for i in range(dim)] + [-np.eye(dim)[i] for i in range(dim)]
        C = np.eye(dim)
        decay = 0.1
        success_steps = []

        while evals < budget:
            improved = False
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
                    improved = True
                    report_best(best_f, best_x)
                    s = step * d
                    success_steps.append(s)
                    if len(success_steps) > 2 * dim:
                        success_steps.pop(0)
                    C = (1 - decay) * C + decay * np.outer(s, s)
                    break
            if improved:
                step *= 1.2
                continue

            n_rand = min(2, dim)
            if success_steps:
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
                f = func(candidate)
                evals += 1
                if f < best_f:
                    best_f = f
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
                if step < 1e-12 * initial_step:
                    remaining = budget - evals
                    if remaining > 0:
                        n_restart = min(max(2 * dim, 1), remaining)
                        restart_pts = lhs_points(n_restart)
                        for i in range(n_restart):
                            if evals >= budget:
                                break
                            x = restart_pts[i]
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