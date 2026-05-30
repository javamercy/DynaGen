import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial mean
        mean = lb + (ub - lb) * self.rng.rand(dim)
        sigma = 0.2 * (ub - lb)  # initial step size
        # covariance matrix (diagonal for simplicity)
        C = np.eye(dim)
        # evolution path for step-size adaptation
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        # strategy parameters
        mu = 4 + int(3 * np.log(dim))  # population size
        lambd = mu  # we set lambda = mu for simplicity
        # weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        # learning rates
        cc = 4.0 / (dim + 4.0)
        cs = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        c1 = 2.0 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2.0 * (mu_eff - 2.0 + 1.0/mu_eff) / ((dim + 2.0)**2 + mu_eff))
        damps = 1.0 + 2.0 * max(0, np.sqrt((mu_eff - 1.0)/(dim + 1.0)) - 1.0) + cs
        # evaluations
        evals = 0
        best_value = None
        best_x = None
        # initial evaluation
        x0 = np.clip(mean, lb, ub)
        f0 = func(x0)
        evals += 1
        best_value = f0
        best_x = x0.copy()
        report_best(best_value, best_x)
        # main loop
        while evals + mu <= self.budget:
            # sample offspring
            samples = []
            for _ in range(mu):
                # sample from N(mean, sigma^2 * C)
                z = self.rng.randn(dim)
                y = mean + sigma * (C @ z)  # C is identity, so C@z = z
                y = np.clip(y, lb, ub)
                samples.append(y)
            # evaluate
            values = []
            for y in samples:
                f = func(y)
                evals += 1
                values.append(f)
                if best_value is None or f < best_value:
                    best_value = f
                    best_x = y.copy()
                    report_best(best_value, best_x)
            # sort
            idx = np.argsort(values)
            samples = [samples[i] for i in idx]
            values = [values[i] for i in idx]
            # update mean
            old_mean = mean.copy()
            mean = np.zeros(dim)
            for i in range(mu):
                mean += weights[i] * samples[i]
            # update evolution paths
            dmean = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (np.linalg.cholesky(C).T @ dmean)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*evals/mu)) < (1.4 + 2.0/(dim+1.0))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * dmean
            # update covariance
            # rank-one update
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            # rank-mu update
            for i in range(mu):
                yi = (samples[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(yi, yi)
            # enforce symmetry (optional)
            C = (C + C.T) / 2
            # update step size
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            # clip sigma?
            sigma = np.clip(sigma, 1e-12, None)
        return best_value, best_x