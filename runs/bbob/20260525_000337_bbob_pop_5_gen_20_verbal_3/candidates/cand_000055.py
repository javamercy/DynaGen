import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
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
        # initial point
        x0 = rng.uniform(lb, ub)
        best_x = x0.copy()
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        lam = min(lam, budget - evals)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = (4.0 + mueff/dim) / (dim + 4.0 + 2.0*mueff/dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0*max(0.0, np.sqrt((mueff-1.0)/(dim+1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3)**1.5 + mueff)
        # state
        m = best_x.copy()
        sigma = (ub - lb).mean() / 5.0
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        no_improve = 0
        max_no_improve = max(30, int(0.2 * budget))
        while evals < budget:
            if no_improve >= max_no_improve:
                # restart
                m = rng.uniform(lb, ub)
                sigma = (ub - lb).mean() / 5.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve = 0
                val = func(m)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = m.copy()
                    report_best(best_val, best_x)
                continue
            lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(dim)
            pop = m + sigma * rng.randn(lam_gen, dim) @ L.T
            pop = np.clip(pop, lb, ub)
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            idx = np.argsort(vals)
            for i in range(lam_gen):
                if vals[i] < best_val:
                    best_val = vals[i]
                    best_x = pop[i].copy()
                    no_improve = 0
                    report_best(best_val, best_x)
                else:
                    no_improve += 1
            x_old = m.copy()
            m = weights @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / sigma
                C_mu += weights[i] * np.outer(z, z)
            C = (1 - ccov) * C + ccov * (np.outer(p_c, p_c) + C_mu)
            C = (C + C.T) / 2.0
            C += 1e-10 * np.eye(dim)
            invL = np.linalg.solve(L, np.eye(dim))
            delta_c = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_c
            norm_ps = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_ps / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
        return best_val, best_x