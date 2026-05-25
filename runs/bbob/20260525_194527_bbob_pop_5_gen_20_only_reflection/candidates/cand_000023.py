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
        # initial mean
        xmean = self.rs.uniform(lb, ub, d)
        fbest = func(xmean)
        xbest = xmean.copy()
        report_best(fbest, xbest)
        total_evals = 1

        # CMA-ES parameters
        sigma_init = 0.3 * np.mean(ub - lb)
        lambda_init = 4 + int(3 * np.log(d))
        restart_factor = 2.0
        max_restarts = 2
        for restart in range(max_restarts + 1):
            lam = int(lambda_init * (restart_factor ** restart))
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
            sigma = sigma_init
            C = np.eye(d)
            norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
            generation = 0
            while total_evals + lam <= self.budget:
                generation += 1
                # Generate offspring with resampling for bounds
                x = np.empty((lam, d))
                fvals = np.empty(lam)
                for i in range(lam):
                    for attempt in range(10):
                        z = self.rs.randn(d)
                        xi = xmean + sigma * np.dot(self.rs.randn(d), np.linalg.cholesky(C).T) if attempt == 0 else xmean + sigma * np.dot(z, np.linalg.cholesky(C).T)
                        if np.all(xi >= lb) and np.all(xi <= ub):
                            break
                        # fallback: uniform in bounds
                        xi = self.rs.uniform(lb, ub, d)
                    x[i] = xi
                    fvals[i] = func(xi)
                    total_evals += 1
                    if fvals[i] < fbest:
                        fbest = fvals[i]
                        xbest = xi.copy()
                        report_best(fbest, xbest)
                # Sort and update
                idx = np.argsort(fvals)
                x_sorted = x[idx]
                old_xmean = xmean.copy()
                xmean = np.dot(weights, x_sorted[:mu])
                x_diff = (xmean - old_xmean) / sigma
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * x_diff
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.solve(np.linalg.cholesky(C), x_diff)
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                diff = x_sorted[:mu] - old_xmean
                diff_norm = diff / sigma
                C += cmu * np.dot(diff_norm.T, np.dot(np.diag(weights), diff_norm))
                C = (C + C.T) / 2.0
                ps_norm = np.linalg.norm(ps)
                sigma = sigma * np.exp((cs / ds) * (ps_norm / norm_expected - 1.0))
                # Restart condition: sigma too small
                if sigma < 0.01 * sigma_init and generation >= 5:
                    break
                # Check if budget left for next generation
                if total_evals + lam > self.budget:
                    break
            # After this run, if restart possible and not last restart, reset mean and keep best
            if restart < max_restarts:
                # Reset mean to best point so far? Usually restart recenters. 
                # But to ensure diversification, we sample a new mean uniformly.
                # But keep best global.
                if total_evals >= self.budget:
                    break
                xmean = self.rs.uniform(lb, ub, d)
                # Optionally re-evaluate? But we already have best.
        return fbest, xbest