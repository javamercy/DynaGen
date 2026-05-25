import numpy as np
from math import log, sqrt, exp

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # initialize
        mean = rng.uniform(lb, ub)
        sigma = 0.3 * (ub - lb).mean()
        C = np.eye(dim)
        A = np.linalg.cholesky(C)  # so that C = A A^T

        best_x = mean.copy()
        best_value = func(mean)
        n_eval = 1
        # report initial
        from report_best import report_best
        report_best(best_value, best_x)

        # CMA parameters
        lambda_ = max(4, int(4 + 3 * log(dim)))
        mu = lambda_ // 2
        weights = np.array([log(mu + 0.5) - log(i+1) for i in range(mu)])
        weights /= weights.sum()
        mueff = 1.0 / (weights**2).sum()
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2.0 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2*max(0, sqrt((mueff-1)/(dim+1)) - 1) + cs

        # evolution path
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # main loop
        while n_eval < budget:
            remaining = budget - n_eval
            if remaining < lambda_:
                # sample remaining points from current distribution
                for _ in range(remaining):
                    z = rng.normal(0, 1, dim)
                    x = mean + sigma * A.dot(z)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    n_eval += 1
                    if val < best_value:
                        best_value = val
                        best_x = x.copy()
                        report_best(best_value, best_x)
                break

            # generate offspring
            arz = rng.normal(0, 1, (lambda_, dim))
            arx = mean + sigma * (A.dot(arz.T)).T
            arx = np.clip(arx, lb, ub)
            arf = np.array([func(x) for x in arx])
            n_eval += lambda_

            # update best
            for i in range(lambda_):
                if arf[i] < best_value:
                    best_value = arf[i]
                    best_x = arx[i].copy()
                    report_best(best_value, best_x)

            # sort by fitness
            idx = np.argsort(arf)
            arz_sorted = arz[idx]
            arx_sorted = arx[idx]

            # update mean
            mean_old = mean.copy()
            mean = np.dot(weights, arx_sorted[:mu])

            # compute natural selection
            zmean = np.dot(weights, arz_sorted[:mu])

            # update evolution paths
            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = (np.linalg.norm(ps) / sqrt(1 - (1 - cs)**(n_eval / lambda_))) < (1.4 + 2.0/(dim+1))
            hsig = float(hsig)
            pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * zmean

            # update covariance matrix
            delta_h = (1 - hsig) * cc * (2 - cc)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_h * C) + cmu * np.dot( (weights * arz_sorted[:mu].T), arz_sorted[:mu] )

            # update step size
            sigma = sigma * exp((cs/damps) * (np.linalg.norm(ps) / sqrt(dim) - 1))

            # recompute Cholesky
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # fallback: ensure positive definite by adding small identity
                C += 1e-9 * np.eye(dim)
                A = np.linalg.cholesky(C)

        return best_value, best_x