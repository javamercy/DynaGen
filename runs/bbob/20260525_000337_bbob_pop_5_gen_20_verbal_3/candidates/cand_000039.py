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
        # Initial point
        best_x = rng.uniform(lb, ub, size=dim)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)
        # CMA-ES parameters (standard)
        lam = 4 + int(2 * np.log(dim))
        lam = max(2, min(lam, budget - evals))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = 4.0 / (dim + 4.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3) ** 1.5)
        # Initialize distribution
        m = best_x.copy()
        sigma = (ub - lb).mean() / 6.0
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        # Restart tracking
        stagnation = int(0.2 * budget)
        no_improve = 0
        while evals < budget:
            if no_improve >= stagnation:
                # Diversity-preserving restart from CMA_ES_Niching
                if rng.rand() < 0.5:
                    # Reinitialize near best with small sigma
                    m = best_x.copy() + 0.1 * sigma * rng.randn(dim)
                    sigma = (ub - lb).mean() / 12.0
                else:
                    # Reinitialize randomly with larger sigma
                    m = rng.uniform(lb, ub, size=dim)
                    sigma = (ub - lb).mean() / 4.0
                m = np.clip(m, lb, ub)
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve = 0
            lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break
            # Sample population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            pop = np.empty((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                pop[i] = m + sigma * L @ z
            np.clip(pop, lb, ub, out=pop)
            # Evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            # Update best
            for i in range(lam_gen):
                if vals[i] < best_val:
                    best_val = vals[i]
                    best_x = pop[i].copy()
                    no_improve = 0
                    report_best(best_val, best_x)
                else:
                    no_improve += 1
            # Sort
            idx = np.argsort(vals)
            # Update mean
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            # Update evolution paths
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            invL = np.linalg.solve(L, np.eye(dim))
            delta_C = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_C
            # Update sigma
            norm_ps = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_ps / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
            # Update covariance
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c)
            C += 1e-10 * np.eye(dim)
        return best_val, best_x