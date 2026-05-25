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
        sigma = 0.3 * np.mean(ub - lb)
        C = np.eye(d)
        lam = 4 + int(3 * np.log(d))
        # parameters
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
        gen = 0
        stagnation_counter = 0
        best_val_gen = fbest
        while total_evals + lam <= self.budget:
            # Check stagnation every floor(10*d/lam) generations
            gen += 1
            if gen >= max(1, 10 * d // lam):
                if fbest < best_val_gen - 1e-10:
                    best_val_gen = fbest
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                gen = 0
                if stagnation_counter >= 2:
                    # Restart: double population, reset CMA parameters
                    lam = min(lam * 2, self.budget - total_evals)
                    if lam < 2:
                        break
                    mu = max(lam // 2, 1)
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= weights.sum()
                    mueff = 1.0 / np.sum(weights**2)
                    cs = (mueff + 2.0) / (d + mueff + 5.0)
                    c1 = 2.0 / ((d + 1.3)**2 + mueff)
                    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
                    ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
                    norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
                    xmean = self.rs.uniform(lb, ub, d)
                    sigma = 0.3 * np.mean(ub - lb)
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)
                    stagnation_counter = 0
                    best_val_gen = fbest
                    if total_evals + lam > self.budget:
                        break
                    continue
            # Sample new points
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-9 * np.eye(d)
                B = np.linalg.cholesky(C)
            z = self.rs.randn(lam, d)
            x_raw = xmean + sigma * np.dot(z, B.T)
            # enforce bounds by resampling (up to 10 tries then clip)
            x = np.empty_like(x_raw)
            for i in range(lam):
                for _ in range(10):
                    if np.all(x_raw[i] >= lb) and np.all(x_raw[i] <= ub):
                        break
                    # resample a new point based on same mean and cov
                    z_new = self.rs.randn(d)
                    x_raw[i] = xmean + sigma * np.dot(B, z_new)
                x[i] = np.clip(x_raw[i], lb, ub)
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
            C = (C + C.T) / 2.0
            ps_norm = np.linalg.norm(ps)
            sigma = sigma * np.exp((cs / ds) * (ps_norm / norm_expected - 1.0))
        return fbest, xbest