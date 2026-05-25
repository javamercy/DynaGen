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
            x = points[i]
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        mean = best_x.copy()
        sigma = 0.2 * np.mean(ub - lb)
        initial_sigma = sigma
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # CMA parameters
        lambd = max(4, int(4 + 3 * np.log(dim)))
        mu = lambd // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1.0/(4*dim) + 1.0/(21*dim**2))

        while evals < budget:
            # Sample candidate solutions
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                A = np.eye(dim)
            z_list = rng.randn(lambd, dim)
            x_list = mean + sigma * (z_list @ A.T)
            x_list = np.clip(x_list, lb, ub)

            # Evaluate
            f_list = np.array([func(x) for x in x_list])
            evals += lambd
            idx = np.argsort(f_list)
            for i in range(lambd):
                if f_list[idx[i]] < best_f:
                    best_f = f_list[idx[i]]
                    best_x = x_list[idx[i]].copy()
                    report_best(best_f, best_x)

            # Update mean
            old_mean = mean.copy()
            x_selected = x_list[idx[:mu]]
            mean = np.dot(weights, x_selected)

            # Update evolution paths
            diff = (mean - old_mean) / sigma
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * diff
            # Compute C^{-1/2} * diff for ps
            try:
                Cinv = np.linalg.inv(C)
                Cinv_half = np.linalg.cholesky(Cinv)
            except np.linalg.LinAlgError:
                Cinv_half = np.linalg.cholesky(np.linalg.inv(C + 1e-12 * np.eye(dim)))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Cinv_half @ diff)

            # Update covariance
            C = (1 - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            # Rank-mu update
            zm = (x_selected - old_mean) / sigma
            for i in range(mu):
                C += cmu * weights[i] * np.outer(zm[i], zm[i])
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)

            # Update step size
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

            # Restart if step size too small or stagnation
            if sigma < 1e-12 * initial_sigma:
                remaining = budget - evals
                if remaining <= 0:
                    break
                n_restart = min(remaining, max(2 * dim, 5))
                points = np.zeros((n_restart, dim))
                for i in range(dim):
                    perm = rng.permutation(n_restart)
                    u = rng.rand(n_restart)
                    points[:, i] = (perm + u) / n_restart
                points = lb + points * (ub - lb)
                for i in range(n_restart):
                    if evals >= budget:
                        break
                    x = points[i]
                    f = func(x)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                mean = best_x.copy()
                sigma = initial_sigma
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)

        return best_f, best_x