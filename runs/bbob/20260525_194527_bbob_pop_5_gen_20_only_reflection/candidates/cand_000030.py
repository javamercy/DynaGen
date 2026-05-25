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
        # initial point
        xmean = self.rs.uniform(lb, ub, d)
        fbest = func(xmean)
        xbest = xmean.copy()
        report_best(fbest, xbest)
        total_evals = 1
        
        # CMA-ES parameters (standard)
        lam0 = 4 + int(3 * np.log(d))
        mu = lam0 // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = 4.0 / (d + 4.0)
        cs = (mueff + 2.0) / (d + mueff + 5.0)
        c1 = 2.0 / ((d + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
        norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
        
        # restart parameters
        max_restarts = 10
        stgn_gen = 10 + d // 5  # generations without improvement trigger restart
        
        for restart in range(max_restarts):
            lam = lam0 * (2 ** restart)
            if total_evals + lam > self.budget:
                break
            sigma = 0.3 * np.mean(ub - lb)
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            xmean = self.rs.uniform(lb, ub, d)
            fmean = func(xmean)
            total_evals += 1
            if fmean < fbest:
                fbest = fmean
                xbest = xmean.copy()
                report_best(fbest, xbest)
            best_in_run = fmean
            no_improve = 0
            
            while total_evals + lam <= self.budget:
                # decompose C
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                # sample
                z = self.rs.randn(lam, d)
                x = xmean + sigma * np.dot(z, B.T)
                # bound handling: resample with fallback to clip
                for i in range(lam):
                    attempts = 0
                    while (np.any(x[i] < lb) or np.any(x[i] > ub)) and attempts < 100:
                        z_i = self.rs.randn(d)
                        x[i] = xmean + sigma * np.dot(B, z_i)
                        attempts += 1
                    x[i] = np.clip(x[i], lb, ub)
                # evaluate
                fvals = np.zeros(lam)
                for i in range(lam):
                    fvals[i] = func(x[i])
                    total_evals += 1
                    if fvals[i] < fbest:
                        fbest = fvals[i]
                        xbest = x[i].copy()
                        report_best(fbest, xbest)
                # check for improvement
                if np.min(fvals) < best_in_run - 1e-12:
                    best_in_run = np.min(fvals)
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= stgn_gen:
                        break
                # update mean and CMA
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
        return fbest, xbest