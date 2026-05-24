import numpy as np
import math

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
        # Evaluate initial point
        x0 = rng.uniform(lb, ub, size=dim)
        best_x = x0.copy()
        best_value = func(best_x)
        evals = 1
        report_best(best_value, best_x)
        # CMA-ES parameters
        lam = 4 + int(3 * math.log(dim))
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
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3)**2)
        # initial state
        sigma0 = (ub - lb).mean() / 6.0
        m = best_x.copy()
        sigma = sigma0
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        restarts = 0
        max_restarts = 3
        initial_lam = lam
        # main loop
        while evals < budget:
            lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                # Evaluate any remaining point
                break
            # sample
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            pop = np.zeros((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                pop[i] = m + sigma * L @ z
            # Reflect coordinates out of bounds
            pop_reflected = np.where(pop < lb, 2*lb - pop, np.where(pop > ub, 2*ub - pop, pop))
            # Ensure within bounds (should be, but clip for safety)
            pop_reflected = np.clip(pop_reflected, lb, ub)
            pop = pop_reflected
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
            p_c = (1 - cc) * p_c + math.sqrt(cc * (2 - cc) * mueff) * delta
            # update p_s
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + math.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            # update sigma
            norm_p_s = np.linalg.norm(p_s)
            sigma *= math.exp((cs / damps) * (norm_p_s / math.sqrt(dim) - 1.0))
            # update C
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c)
            C += 1e-15 * np.eye(dim)  # ensure positive definiteness
            # Restart condition: stagnation or diversity loss
            restart_cond = (sigma < 1e-12 * sigma0) or (np.max(np.linalg.eigvalsh(C)) < 1e-12)
            if restart_cond and restarts < max_restarts and evals < budget - 10:
                restarts += 1
                # Increase population size
                lam = int(initial_lam * (2 ** restarts))
                lam = min(lam, budget - evals)
                if lam < 2:
                    lam = 2
                mu = lam // 2
                if mu < 1:
                    mu = 1
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
                w = w / w.sum()
                mueff = 1.0 / np.sum(w**2)
                # Reinitialize
                m = rng.uniform(lb, ub, size=dim)
                val = func(m)
                evals += 1
                if val < best_value:
                    best_value = val
                    best_x = m.copy()
                    report_best(best_value, best_x)
                sigma = sigma0 * (1 + 0.5 * restarts)
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
        return best_value, best_x