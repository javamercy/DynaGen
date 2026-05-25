import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rs = np.random.RandomState(seed)

    def __call__(self, func):
        d = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial evaluation
        xmean = self.rs.uniform(lb, ub, d)
        fbest = func(xmean)
        xbest = xmean.copy()
        report_best(fbest, xbest)
        total_evals = 1
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(d))
        inc_factor = 2.0
        max_restarts = 5
        restart = 0
        while restart <= max_restarts and total_evals < self.budget:
            if restart > 0:
                lam = int(lam * inc_factor)
                # reinitialize mean near best point with noise
                sigma_init = 0.3 * np.mean(ub - lb)
                xmean = xbest + 0.1 * sigma_init * self.rs.randn(d)
                xmean = np.clip(xmean, lb, ub)
                f = func(xmean)
                total_evals += 1
                if f < fbest:
                    fbest = f
                    xbest = xmean.copy()
                    report_best(fbest, xbest)
            sigma = 0.3 * np.mean(ub - lb)
            C = np.eye(d)
            mu = max(lam // 2, 1)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = 4.0 / (d + 4.0)
            cs = (mueff + 2.0) / (d + mueff + 5.0)
            c1 = 2.0 / ((d + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
            pc = np.zeros(d)
            ps = np.zeros(d)
            norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
            staleness = 0
            best_in_run = fbest
            gen = 0
            while total_evals + lam <= self.budget:
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                z = self.rs.randn(lam, d)
                x = xmean + sigma * np.dot(z, B.T)
                x = np.clip(x, lb, ub)
                fvals = np.zeros(lam)
                for i in range(lam):
                    fvals[i] = func(x[i])
                    total_evals += 1
                    if fvals[i] < fbest:
                        fbest = fvals[i]
                        xbest = x[i].copy()
                        report_best(fbest, xbest)
                idx = np.argsort(fvals)
                x_sorted = x[idx]
                old_xmean = xmean.copy()
                xmean = np.dot(weights, x_sorted[:mu])
                x_diff = (xmean - old_xmean) / sigma
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * x_diff
                try:
                    invB = np.linalg.solve(B, np.eye(d))
                except np.linalg.LinAlgError:
                    invB = np.linalg.inv(B)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invB, x_diff)
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                diff = x_sorted[:mu] - old_xmean
                diff_norm = diff / sigma
                C += cmu * np.dot(diff_norm.T, np.dot(np.diag(weights), diff_norm))
                C = (C + C.T) / 2
                ps_norm = np.linalg.norm(ps)
                sigma = sigma * np.exp((cs / ds) * (ps_norm / norm_expected - 1.0))
                if fvals[idx[0]] < best_in_run:
                    best_in_run = fvals[idx[0]]
                    staleness = 0
                else:
                    staleness += 1
                gen += 1
                max_stall = max(10, 5 + 3 * d // max(lam, 1))
                if staleness > max_stall or sigma < 1e-12:
                    break
            if fbest < best_in_run:
                best_in_run = fbest
            restart += 1
        return fbest, xbest