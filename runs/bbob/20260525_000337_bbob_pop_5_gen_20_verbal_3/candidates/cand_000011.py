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
        best_x = rng.uniform(lb, ub, size=dim)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)
        # ES parameters
        lam = 4 + int(3 * np.log(dim))
        lam = min(lam, budget - evals)
        if lam < 2:
            lam = 2
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # CSA parameters
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        # state
        m = best_x.copy()
        sigma = (ub - lb).mean() / 6.0
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
                pop[i] = m + sigma * z
            np.clip(pop, lb, ub, out=pop)
            # evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            # update best
            for i in range(lam_gen):
                if vals[i] < best_val:
                    best_val = vals[i]
                    best_x = pop[i].copy()
                    report_best(best_val, best_x)
            # sort
            idx = np.argsort(vals)
            # update mean
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            # update p_s
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta
            # update sigma
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
        return best_val, best_x