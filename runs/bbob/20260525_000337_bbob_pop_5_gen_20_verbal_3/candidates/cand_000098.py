import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = self.rng
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        # Initial point
        best_x = rng.uniform(lb, ub, size=dim)
        best_value = func(best_x)
        evals = 1
        report_best(best_value, best_x)
        # CMA-ES parameters (standard)
        lam = 4 + int(3 * np.log(dim))
        lam = min(lam, budget - evals)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = (1.0 / mueff) * (2.0 / (dim + 1.414) ** 2) + (1.0 - 1.0 / mueff) * (1.0 / ((dim + 1.414) ** 2))
        ccov = min(1.0, ccov)
        # State
        m = best_x.copy()
        sigma = 0.3 * (ub - lb).mean()
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        no_improve_evals = 0
        restart_threshold = int(0.2 * budget)
        while evals < budget:
            if no_improve_evals >= restart_threshold:
                # Restart
                m = rng.uniform(lb, ub, size=dim)
                sigma = 0.3 * (ub - lb).mean()
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
            # Sample
            A = np.linalg.cholesky(C)
            Z = rng.randn(lam, dim)
            pop = m + sigma * (Z @ A.T)
            pop = np.clip(pop, lb, ub)
            vals = np.array([func(p) for p in pop])
            evals += lam
            # Update best
            for i in range(lam):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    no_improve_evals = 0
                    report_best(best_value, best_x)
                else:
                    no_improve_evals += 1
            # Selection
            idx = np.argsort(vals)
            x_old = m.copy()
            m = weights @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            # Step-size adaptation
            invsqrtC = np.linalg.solve(A, np.eye(dim))
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ delta)
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
            # Covariance adaptation
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            Cmu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / sigma
                Cmu += weights[i] * np.outer(z, z)
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov * (1 - 1 / mueff) * Cmu
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(dim)
        return best_value, best_x