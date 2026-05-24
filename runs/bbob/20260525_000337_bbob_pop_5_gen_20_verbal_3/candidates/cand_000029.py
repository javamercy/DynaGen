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
        bounds_range = ub - lb
        x0 = rng.uniform(lb, ub, size=dim)
        best_x = x0.copy()
        best_value = func(best_x)
        evals = 1
        report_best(best_value, best_x)
        # CMA-ES parameters
        lam = 4 + int(2 * np.log(dim))
        lam = max(2, min(lam, budget - evals))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = 4.0 / (dim + 4.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3)**1.5)
        m = best_x.copy()
        sigma = bounds_range.mean() / 6.0
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        no_improve_evals = 0
        restart_threshold = int(0.2 * budget)
        while evals < budget:
            if no_improve_evals >= restart_threshold:
                # Exploration: larger perturbation and sigma
                pert = rng.uniform(-0.5, 0.5, size=dim) * bounds_range
                m = np.clip(best_x + pert, lb, ub)
                sigma = bounds_range.mean() * 0.8
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
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
                    no_improve_evals = 0
                    report_best(best_value, best_x)
                    improved = True
            if not improved:
                no_improve_evals += lam_gen
            idx = np.argsort(vals)
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            try:
                invL = np.linalg.solve(L, np.eye(dim))
            except np.linalg.LinAlgError:
                invL = np.eye(dim)
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c)
            C += 1e-10 * np.eye(dim)
        return best_value, best_x