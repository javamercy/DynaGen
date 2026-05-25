import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        d = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rs = np.random.RandomState(self.seed)
        # initial point
        xmean = rs.uniform(lb, ub, d)
        fbest = func(xmean)
        xbest = xmean.copy()
        report_best(fbest, xbest)
        total_evals = 1
        # CMA parameters
        sigma0 = 0.3 * np.mean(ub - lb)
        lam = 4 + int(3 * np.log(d))
        mu = max(lam // 2, 1)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = 4.0 / (d + 4.0)
        cs = (mueff + 2.0) / (d + mueff + 5.0)
        c1 = 2.0 / ((d + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
        norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
        sigma = sigma0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        no_improve_gen = 0
        while total_evals + lam <= self.budget:
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-9 * np.eye(d)
                B = np.linalg.cholesky(C)
            x = np.zeros((lam, d))
            for i in range(lam):
                for _ in range(100):
                    z = rs.randn(d)
                    candidate = xmean + sigma * np.dot(B, z)
                    if np.all(candidate >= lb) and np.all(candidate <= ub):
                        break
                else:
                    candidate = np.clip(candidate, lb, ub)
                x[i] = candidate
            fvals = np.empty(lam)
            for i in range(lam):
                fvals[i] = func(x[i])
                total_evals += 1
                if fvals[i] < fbest:
                    fbest = fvals[i]
                    xbest = x[i].copy()
                    report_best(fbest, xbest)
                    no_improve_gen = 0
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
            C = (1 - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            diff = x_sorted[:mu] - old_xmean
            diff_norm = diff / sigma
            C += cmu * np.dot(diff_norm.T, np.dot(np.diag(weights), diff_norm))
            C = (C + C.T) / 2.0
            ps_norm = np.linalg.norm(ps)
            sigma = sigma * np.exp((cs / ds) * (ps_norm / norm_expected - 1.0))
            no_improve_gen += 1
            restart_after = max(10, int(30.0 * d / lam))
            if no_improve_gen >= restart_after:
                new_lam = 2 * lam
                if total_evals + new_lam <= self.budget:
                    lam = new_lam
                    mu = max(lam // 2, 1)
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= weights.sum()
                    mueff = 1.0 / np.sum(weights**2)
                    cs = (mueff + 2.0) / (d + mueff + 5.0)
                    c1 = 2.0 / ((d + 1.3)**2 + mueff)
                    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
                    ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
                    sigma = sigma0
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)
                    # reinitialize near best point with small noise
                    xmean = xbest + sigma0 * 0.5 * rs.randn(d)
                    xmean = np.clip(xmean, lb, ub)
                    no_improve_gen = 0
                else:
                    break
        return fbest, xbest