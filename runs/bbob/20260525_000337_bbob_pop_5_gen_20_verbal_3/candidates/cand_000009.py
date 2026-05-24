import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        spans = ub - lb
        mean_span = np.mean(spans)
        calls = 0
        best_val = np.inf
        best_x = None

        def evaluate(x):
            nonlocal calls, best_val, best_x
            x = np.clip(x, lb, ub)
            val = func(x)
            calls += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            return val

        # Initial point
        x0 = rng.uniform(lb, ub)
        evaluate(x0)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        if lam < 2:
            lam = 2
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = 4.0 / (dim + 4.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3) ** 2)

        while calls < budget:
            # Restart from random point
            m = rng.uniform(lb, ub)
            if calls < budget:
                evaluate(m)
            else:
                break
            sigma = mean_span / 6.0
            C = np.eye(dim)
            p_c = np.zeros(dim)
            p_s = np.zeros(dim)
            stagnation = 0
            while calls < budget:
                lam_gen = min(lam, budget - calls)
                if lam_gen < 2:
                    break
                # Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                # Sample
                pop = np.zeros((lam_gen, dim))
                z = rng.randn(lam_gen, dim)
                for i in range(lam_gen):
                    pop[i] = m + sigma * L @ z[i]
                pop = np.clip(pop, lb, ub)
                # Evaluate
                vals = np.array([func(p) for p in pop])
                calls += lam_gen
                # Update best
                for i in range(lam_gen):
                    if vals[i] < best_val:
                        best_val = vals[i]
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                # Sort
                idx = np.argsort(vals)
                # Update mean
                x_old = m.copy()
                m = weights @ pop[idx[:mu]]
                delta = (m - x_old) / sigma
                # Update p_c
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
                # Update p_s
                invL = np.linalg.solve(L, np.eye(dim))
                delta_Cinv = invL @ delta
                p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
                # Update sigma
                norm_p_s = np.linalg.norm(p_s)
                sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
                # Update C
                C = (1 - ccov) * C + ccov * np.outer(p_c, p_c)
                C += 1e-10 * np.eye(dim)
                # Check stagnation
                if calls < budget:
                    # improvement check: best_val unchanged from previous generation?
                    # We track if best_val improved in this generation
                    if np.any(vals < best_val):  # actually best_val already updated, so check if any new best found
                        stagnation = 0
                    else:
                        stagnation += 1
                if sigma < 1e-8 * mean_span or stagnation > 10 + dim:
                    break
        return best_val, best_x