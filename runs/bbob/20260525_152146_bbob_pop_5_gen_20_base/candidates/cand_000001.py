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
        # initialize sigma as fraction of average range
        sigma = 0.2 * np.mean(ub - lb)
        # population size
        lam = int(4 + 3 * np.log(dim))
        mu = lam // 2
        # recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        # adaptation parameters
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(dim+1)) - 1) + cs
        # initialize dynamic state
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        invsqrtC = np.eye(dim)
        eigeneval = 0
        # evaluate initial mean
        best_x = mean.copy()
        best_val = func(best_x)
        report_best(best_val, best_x)
        calls = 1
        # main loop
        while calls < budget:
            # sample new population
            mean_old = mean.copy()
            pop = np.empty((lam, dim))
            fit = np.empty(lam)
            for i in range(lam):
                z = rng.randn(dim)
                pop[i] = mean + sigma * (B @ (D * z))
                # clip to bounds
                pop[i] = np.clip(pop[i], lb, ub)
                fit[i] = func(pop[i])
                calls += 1
                if fit[i] < best_val:
                    best_val = fit[i]
                    best_x = pop[i].copy()
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
            # update evolution paths
            dmean = mean_new - mean_old
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ np.linalg.solve(B, dmean))
            # clipping for numerical stability
            ps = np.clip(ps, -1e100, 1e100)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*calls/lam)) < 1.4 + 2/(dim+1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * dmean
            # update covariance matrix
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc)) + cmu * (pop[:mu] - mean_old).T @ np.diag(weights) @ (pop[:mu] - mean_old) / sigma**2
            C = (C + C.T) / 2  # enforce symmetry
            # update step size
            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/np.sqrt(dim) - 1))
            sigma = max(sigma, 1e-10)
            # update eigen decomposition if needed
            if calls - eigeneval > lam / (c1+cmu)/dim/10:
                eigeneval = calls
                C = (C + C.T) / 2
                try:
                    D, B = np.linalg.eigh(C)
                except np.linalg.LinAlgError:
                    D = np.ones(dim)
                    B = np.eye(dim)
                D = np.sqrt(np.maximum(D, 1e-20))
                invsqrtC = B @ np.diag(1/D) @ B.T
            # set mean for next iteration
            mean = mean_new
        return best_val, best_x