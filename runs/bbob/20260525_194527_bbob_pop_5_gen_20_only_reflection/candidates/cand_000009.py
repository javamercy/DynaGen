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
        rang = ub - lb

        # initial mean
        xmean = self.rs.uniform(lb, ub, d)
        fbest = func(xmean)
        xbest = xmean.copy()
        report_best(fbest, xbest)
        total_evals = 1

        # population sizes
        lam = max(4, min(10, 4 + int(3 * np.log(d))))
        mu = lam // 2
        if mu < 1:
            mu = 1
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # adaptation params
        cs = (mueff + 2.0) / (d + mueff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
        # per-dimension step sizes and evolution paths
        sigma = 0.3 * rang
        ps = np.zeros(d)

        norm_expected = 1.0  # for 1D, expected norm of ps is approximately 1? Use formula from CMA-ES for d=1
        # Actually for d=1, expected norm = sqrt(1)*(1 - 1/(4*1) + 1/(21*1^2)) = 0.914
        norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))

        while total_evals + lam <= self.budget:
            # sample candidates
            z = self.rs.randn(lam, d)
            candidates = xmean + sigma * z
            candidates = np.clip(candidates, lb, ub)

            # evaluate
            fvals = np.zeros(lam)
            for i in range(lam):
                fvals[i] = func(candidates[i])
                total_evals += 1
                if fvals[i] < fbest:
                    fbest = fvals[i]
                    xbest = candidates[i].copy()
                    report_best(fbest, xbest)

            # sort and get best mu
            idx = np.argsort(fvals)
            x_sorted = candidates[idx]
            old_xmean = xmean.copy()
            xmean = np.dot(weights, x_sorted[:mu])

            # update evolution path per dimension
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (xmean - old_xmean) / sigma
            # update step sizes
            sigma = sigma * np.exp((cs / ds) * (np.abs(ps) - norm_expected))
            # keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-12 * rang, 0.5 * rang)

        return fbest, xbest