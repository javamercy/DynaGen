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
        best_val = np.inf
        best_x = None
        total_evals = 0
        # initial point
        x0 = self.rs.uniform(lb, ub, d)
        f0 = func(x0)
        total_evals += 1
        best_val = f0
        best_x = x0.copy()
        report_best(best_val, best_x)
        # CMA-ES parameters
        sigma_init = 0.3 * np.mean(ub - lb)
        lam = 4 + int(3 * np.log(d))
        # restart loop
        restart = 0
        while total_evals < self.budget:
            sigma = sigma_init
            C = np.eye(d)
            xmean = x0.copy()
            pc = np.zeros(d)
            ps = np.zeros(d)
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = 4.0 / (d + 4.0)
            cs = (mueff + 2.0) / (d + mueff + 5.0)
            c1 = 2.0 / ((d + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
            norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
            gen = 0
            no_improve = 0
            while total_evals + lam <= self.budget:
                # sample
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                z = self.rs.randn(lam, d)
                x = xmean + sigma * np.dot(z, B.T)
                # bound handling: resample if outside
                for i in range(lam):
                    for _ in range(10):  # try up to 10 times
                        if np.all(x[i] >= lb) and np.all(x[i] <= ub):
                            break
                        z_i = self.rs.randn(d)
                        x[i] = xmean + sigma * np.dot(B, z_i)
                    else:
                        # fallback clipping
                        x[i] = np.clip(x[i], lb, ub)
                fvals = np.zeros(lam)
                for i in range(lam):
                    fvals[i] = func(x[i])
                    total_evals += 1
                    if fvals[i] < best_val:
                        best_val = fvals[i]
                        best_x = x[i].copy()
                        report_best(best_val, best_x)
                        no_improve = 0
                    else:
                        no_improve += 1
                # sort
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
                C = (C + C.T) / 2.0
                ps_norm = np.linalg.norm(ps)
                sigma = sigma * np.exp((cs / ds) * (ps_norm / norm_expected - 1.0))
                gen += 1
                # stagnation check: no improvement for large period or sigma collapse
                if no_improve > 100 + 4 * lam or sigma < 1e-10 * np.mean(ub - lb):
                    break
            # restart: double population, reinitialize mean
            restarts += 1
            lam = lam * 2
            mu = lam // 2
            if mu == 0:
                mu = 1
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            # reinitialize x0 randomly but keep best
            x0 = self.rs.uniform(lb, ub, d)
            sigma_init = 0.3 * np.mean(ub - lb)
        return best_val, best_x