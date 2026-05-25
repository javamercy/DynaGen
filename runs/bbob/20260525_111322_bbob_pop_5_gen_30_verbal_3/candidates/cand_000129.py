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

        # Parameters for evolution strategy
        initial_sigma = 0.2 * np.mean(ub - lb)
        sigma = initial_sigma
        C = np.eye(dim)
        invsqrtC = np.eye(dim)
        p_c = np.zeros(dim)
        p_sigma = np.zeros(dim)
        use_success_steps = []
        max_steps = 2 * dim

        # CMA-ES parameters (approximate for single-point update)
        c_sigma = 1.0 / (2 * dim)
        d_sigma = 1.0
        c1 = 2.0 / (dim ** 2)
        cmu = min(1 - c1, 2.0 * (dim + 1) / (dim ** 2 + 4))
        decay = 0.1

        chiN = np.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim*dim))

        def update_invsqrtC():
            nonlocal invsqrtC
            try:
                L = np.linalg.cholesky(C + 1e-12*np.eye(dim))
                invsqrtC = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                invsqrtC = np.eye(dim)

        update_invsqrtC()

        while evals < budget:
            improved = False
            old_best_x = best_x.copy()

            # Sample directions: coordinate and random
            directions = []
            for i in range(dim):
                directions.append(np.eye(dim)[i])
                directions.append(-np.eye(dim)[i])
            n_rand = min(3, dim)
            for _ in range(n_rand):
                z = rng.randn(dim)
                L = np.linalg.cholesky(C + 1e-12*np.eye(dim))
                rand_dir = L @ z
                norm = np.linalg.norm(rand_dir)
                if norm > 0:
                    directions.append(rand_dir / norm)
                else:
                    directions.append(np.zeros(dim))

            for d in directions:
                if evals >= budget:
                    break
                candidate = best_x + sigma * d
                candidate = np.clip(candidate, lb, ub)
                f = func(candidate)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    improved = True
                    step = best_x - old_best_x
                    # Update evolution paths and covariance
                    p_c = (1 - c1) * p_c + np.sqrt(c1 * (2 - c1)) * (step / sigma)
                    z = invsqrtC @ (step / sigma)
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * z
                    sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chiN - 1))
                    sigma = max(sigma, 1e-12 * initial_sigma)
                    C = (1 - decay) * C + decay * (c1 * np.outer(p_c, p_c))
                    if len(use_success_steps) > 0:
                        rank_mu = np.zeros((dim, dim))
                        for s in use_success_steps:
                            rank_mu += np.outer(s, s)
                        rank_mu /= len(use_success_steps)
                        C += decay * cmu * rank_mu
                    C = (C + C.T) / 2
                    C += 1e-12 * np.eye(dim)
                    use_success_steps.append(step)
                    if len(use_success_steps) > max_steps:
                        use_success_steps.pop(0)
                    update_invsqrtC()
                    break

            if not improved:
                sigma *= 0.9
                if sigma < 1e-12 * initial_sigma:
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
                        sigma = initial_sigma
                        C = np.eye(dim)
                        p_c = np.zeros(dim)
                        p_sigma = np.zeros(dim)
                        use_success_steps.clear()
                        update_invsqrtC()

        return best_f, best_x