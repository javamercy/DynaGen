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
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub
        # initial mean uniform in bounds
        mean = lb + rng.rand(dim) * (ub - lb)
        # global step size
        sigma = 0.2 * np.mean(ub - lb)
        # per-dimension step sizes (initialized to 1)
        per_dim_sigma = np.ones(dim)
        # population size
        lam = int(4 + 3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        # adaptation parameters (simplified: use CSA for global sigma, per-dim adaptation from ES)
        cs = (mueff + 2) / (dim + mueff + 5)
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(dim+1)) - 1) + cs
        # cumulative path for global step size
        ps = np.zeros(dim)
        # learning rate for per-dim sigma update (from Sep-CMA-ES)
        c_d = (1 + dim/float(mueff)) / (2 * dim + 1)  # typical value
        # evaluate initial mean
        best_x = mean.copy()
        best_val = func(best_x)
        report_best(best_val, best_x)
        calls = 1
        # main loop
        while calls < budget:
            mean_old = mean.copy()
            # sample population
            pop = np.empty((lam, dim))
            fit = np.empty(lam)
            for i in range(lam):
                z = rng.randn(dim)  # standard normal
                # sample with per-dim sigma and global sigma
                sample = mean + sigma * per_dim_sigma * z
                sample = np.clip(sample, lb, ub)
                pop[i] = sample
                fit[i] = func(sample)
                calls += 1
                if fit[i] < best_val:
                    best_val = fit[i]
                    best_x = sample.copy()
                    report_best(best_val, best_x)
                if calls >= budget:
                    break
            if calls >= budget:
                break
            # sort by fitness
            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]
            # update mean
            mean_new = np.zeros(dim)
            for i in range(mu):
                mean_new += weights[i] * pop[i]
            # update global step size via cumulative path
            # we need to compute the path using the difference in mean
            dmean = (mean_new - mean_old) / sigma  # normalized difference
            # but for path we need to account for per-dim scaling? Simplified: use isotropy
            # Actually, Sep-CMA-ES uses a separate path for each dimension? We'll use CSA on the standardized difference.
            # We'll compute the overall step using the mean difference scaled by per_dim_sigma
            dmean_scaled = dmean / per_dim_sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * dmean_scaled
            # clip ps to avoid numerical issues
            ps = np.clip(ps, -1e100, 1e100)
            # update global sigma
            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/np.sqrt(dim) - 1))
            sigma = max(sigma, 1e-10)
            # update per-dim sigma based on empirical variance of selected offspring
            # compute weighted empirical variance
            var = np.zeros(dim)
            for i in range(mu):
                diff = pop[i] - mean_old
                var += weights[i] * diff**2
            # normalize by global sigma^2
            var = var / (sigma**2 + 1e-20)
            # update per_dim_sigma with learning rate
            per_dim_sigma = per_dim_sigma * np.exp( (c_d/2) * (var - 1) )
            per_dim_sigma = np.clip(per_dim_sigma, 1e-10, 1e10)
            # set mean for next iteration
            mean = mean_new
        return best_val, best_x