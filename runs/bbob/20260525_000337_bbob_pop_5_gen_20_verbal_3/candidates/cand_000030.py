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
        x0 = rng.uniform(lb, ub, size=dim)
        best_x = x0.copy()
        best_value = func(best_x)
        evals = 1
        report_best(best_value, best_x)
        lam = 4 + int(2 * np.log(dim))
        lam = min(lam, budget - evals)
        if lam < 2:
            lam = 2
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = 4.0 / (dim + 4.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 1.0 / np.sqrt(dim)
        m = best_x.copy()
        sigma = (ub - lb).mean() / 5.0  # smaller initial sigma
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        last_improvement_evals = 0
        while evals < budget:
            lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            pop = np.zeros((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                pop[i] = m + sigma * L @ z
            np.clip(pop, lb, ub, out=pop)
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            improved = False
            for i in range(lam_gen):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    report_best(best_value, best_x)
                    improved = True
                    last_improvement_evals = evals
            idx = np.argsort(vals)
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c)
            C += 1e-10 * np.eye(dim)
            # Local search if no improvement in this generation
            if not improved:
                local_evals = min(dim + 1, budget - evals)
                if local_evals > 0:
                    local_sigma = sigma * 0.1
                    for _ in range(local_evals):
                        z = rng.randn(dim)
                        x = best_x + local_sigma * z
                        np.clip(x, lb, ub, out=x)
                        val = func(x)
                        evals += 1
                        if val < best_value:
                            best_value = val
                            best_x = x.copy()
                            report_best(best_value, best_x)
                            last_improvement_evals = evals
        return best_value, best_x