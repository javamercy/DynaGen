import numpy as np
import math
from functools import reduce

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * math.log(dim))  # population size
        mu = lambda_ // 2
        weights = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        weights /= weights.sum()  # normalize
        mueff = 1.0 / (weights**2).sum()  # variance effectiveness

        # Strategy parameters: default from CMA-ES
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2*max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs

        # Initialize
        x_mean = rng.uniform(lb, ub, size=dim)
        sigma = 0.5 * (ub - lb).mean()  # initial step-size
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)

        # Track best
        best_x = x_mean.copy()
        f = func(best_x)
        best_val = f
        report_best(best_val, best_x)
        n_evals = 1

        # Main loop
        while n_evals < budget:
            # Sample population
            arz = rng.randn(dim, lambda_)
            arx = np.zeros((dim, lambda_))
            for k in range(lambda_):
                arx[:, k] = x_mean + sigma * (B @ (D * arz[:, k]))
            # Clip
            arx = np.clip(arx, lb[:, None], ub[:, None])

            # Evaluate
            fvals = np.empty(lambda_)
            for k in range(lambda_):
                if n_evals + k >= budget:
                    # Not enough budget for full population
                    lambda_eff = budget - n_evals
                    arx = arx[:, :lambda_eff]
                    fvals = fvals[:lambda_eff]
                    break
                fvals[k] = func(arx[:, k])
                # Update best
                if fvals[k] < best_val:
                    best_val = fvals[k]
                    best_x = arx[:, k].copy()
                    report_best(best_val, best_x)
            n_evals += len(fvals)

            if len(fvals) == 0:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            arz_sorted = arz[:, idx]
            arx_sorted = arx[:, idx]

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.dot(arx_sorted[:, :mu], weights)

            # Update evolution paths
            invsqrtC = B @ np.diag(1.0 / D) @ B.T
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ ((x_mean - x_old) / sigma)
            hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**(2*n_evals/lambda_))) < (1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * ((x_mean - x_old) / sigma)

            # Update covariance
            art = arz_sorted[:, :mu]
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * C) + cmu * np.dot((art * weights), art.T)

            # Update step-size
            sigma *= math.exp((cs/damps) * (np.linalg.norm(ps)/math.sqrt(dim) - 1))

            # Update B and D from C
            C = (C + C.T) / 2  # ensure symmetry
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 0))  # ensure non-negative

            # Check for stagnation
            if sigma < 1e-12:
                break

        return best_val, best_x