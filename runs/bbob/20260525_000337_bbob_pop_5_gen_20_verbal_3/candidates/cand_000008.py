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
        # Evaluate initial point
        x0 = rng.uniform(lb, ub, size=dim)
        best_x = x0.copy()
        best_value = func(best_x)
        evals = 1
        report_best(best_value, best_x)
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        lam = min(lam, budget - evals)
        if lam < 2:
            lam = 2
        mu = lam // 2
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        # adaptation constants
        cc = 4.0 / (dim + 4.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3)**2)
        # initial state
        m = best_x.copy()
        sigma = (ub - lb).mean() / 6.0
        D = np.ones(dim)  # diagonal variances
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        # main loop
        while evals < budget:
            lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break
            # sample
            pop = np.zeros((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                pop[i] = m + sigma * np.sqrt(D) * z
            np.clip(pop, lb, ub, out=pop)
            # evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            # update best
            for i in range(lam_gen):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    report_best(best_value, best_x)
            # sort
            idx = np.argsort(vals)
            # update mean
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            # update p_c
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            # update D (rank-one update)
            D = (1 - ccov) * D + ccov * (p_c**2)
            D = np.maximum(D, 1e-10)  # avoid zero
            # update p_s
            inv_sqrt_D = 1.0 / np.sqrt(D + 1e-20)
            delta_norm = delta * inv_sqrt_D
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_norm
            # update sigma
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
        return best_value, best_x