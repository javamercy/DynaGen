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

        # Parameters
        initial_step = 0.2 * np.mean(ub - lb)
        step = initial_step
        min_step = 1e-12 * initial_step
        max_step = 0.5 * np.mean(ub - lb)
        # Coordinate directions
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)
        # Covariance and evolution path
        C = np.eye(dim)
        p_c = np.zeros(dim)
        decay = 0.1
        c_cov = 0.2
        c_p = 0.2
        success_steps = []
        # Success rate tracking
        window_size = 20
        success_count = 0
        total_count = 0
        explore_prob = 0.2

        while evals < budget:
            # Exploration random point
            if rng.rand() < explore_prob and evals < budget:
                candidate = lb + rng.rand(dim) * (ub - lb)
                f_val = func(candidate)
                evals += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    s = candidate - best_x
                    success_steps.append(s)
                    if len(success_steps) > 2 * dim:
                        success_steps.pop(0)
                    p_c = (1 - c_p) * p_c + c_p * s
                    C = (1 - decay) * C + decay * np.outer(p_c, p_c) + 0.01 * np.outer(s, s)
                    success_count += 1
                    total_count += 1
                    continue
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
                    success_steps.append(s)
                    if len(success_steps) > 2 * dim:
                        success_steps.pop(0)
                    p_c = (1 - c_p) * p_c + c_p * s
                    C = (1 - decay) * C + decay * np.outer(p_c, p_c) + 0.01 * np.outer(s, s)
                    success_count += 1
                    total_count += 1
                    break
            if improved:
                total_count += 1
                # Adjust step based on success rate
                success_rate = success_count / max(1, total_count)
                if success_rate > 0.2:
                    step = min(step * 1.2, max_step)
                else:
                    step = max(step * 0.9, min_step)
                continue
            # Random directions from covariance
            n_rand = min(3, dim)
            if len(success_steps) > 0:
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
                    p_c = (1 - c_p) * p_c + c_p * s
                    C = (1 - decay) * C + decay * np.outer(p_c, p_c) + 0.01 * np.outer(s, s)
                    success_count += 1
                    total_count += 1
                    break
            if improved:
                total_count += 1
                success_rate = success_count / max(1, total_count)
                if success_rate > 0.2:
                    step = min(step * 1.2, max_step)
                else:
                    step = max(step * 0.9, min_step)
            else:
                step *= 0.5
                total_count = 0
                success_count = 0
                if step < min_step:
                    remaining = budget - evals
                    if remaining > 0:
                        n_restart = min(remaining, max(2 * dim, 1))
                        restart_points = np.zeros((n_restart, dim))
                        for i in range(dim):
                            perm = rng.permutation(n_restart)
                            u = rng.rand(n_restart)
                            restart_points[:, i] = (perm + u) / n_restart
                        restart_points = lb + restart_points * (ub - lb)
                        random_point = lb + rng.rand(1, dim) * (ub - lb)
                        restart_points = np.vstack((restart_points, random_point))
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
                        success_steps.clear()
        return best_f, best_x