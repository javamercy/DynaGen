import numpy as np
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Initial point
        best_x = rng.uniform(lb, ub, dim)
        best_val = func(best_x)
        report_best(best_val, best_x)
        evals = 1

        # If budget is 1, just return
        if budget <= 1:
            return best_val, best_x

        # Determine population size
        lambda_ = max(2, min(budget - 1, 4 + int(3 * math.log(dim))))
        # Ensure we can do at least one generation
        if lambda_ > budget - 1:
            lambda_ = budget - 1

        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        # Weighted recombination weights
        weights = np.array([math.log(lambda_ + 0.5) - math.log(i + 1) for i in range(mu)])
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

        # Initialize distribution
        m = best_x.copy()
        sigma = 0.2 * np.mean(ub - lb)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Generation loop
        max_gen = (budget - 1) // lambda_
        for gen in range(max_gen):
            # Sample population
            try:
                X = rng.multivariate_normal(m, sigma ** 2 * C, size=lambda_)
            except:
                C = C + 1e-12 * np.eye(dim)
                X = rng.multivariate_normal(m, sigma ** 2 * C, size=lambda_)
            # Clip to bounds
            X = np.clip(X, lb, ub)
            # Evaluate
            F = np.array([func(x) for x in X])
            evals += lambda_
            # Sort
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]
            # Update best
            if F_sorted[0] < best_val:
                best_val = F_sorted[0]
                best_x = X_sorted[0].copy()
                report_best(best_val, best_x)
            # Update mean
            m_old = m.copy()
            m = np.dot(weights, X_sorted[:mu])
            # Clip mean
            m = np.clip(m, lb, ub)
            # Compute evolution paths
            z = (X_sorted[:mu] - m_old) / sigma
            dvec = np.dot(weights, z)  # weighted mean of z
            ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mueff) * dvec
            hsig = (np.linalg.norm(ps) / math.sqrt(1.0 - (1.0 - cs) ** (2 * (gen + 1)))) < 1.4 + 2.0 / (dim + 1.0)
            pc = (1.0 - cc) * pc + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * dvec
            # Covariance matrix update
            artmp = (X_sorted[:mu] - m_old) / sigma
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C) + cmu * np.dot(weights * artmp.T, artmp)
            # Enforce symmetry
            C = (C + C.T) / 2.0
            # Ensure positive definiteness
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C = C + 1e-12 * np.eye(dim)
            # Step size update
            sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1.0))
            # Check budget
            if evals >= budget:
                break

        # Leftover evaluations: random sampling
        remaining = budget - evals
        while remaining > 0:
            x = rng.uniform(lb, ub, dim)
            val = func(x)
            evals += 1
            remaining -= 1
            if val < best_val:
                best_val = val
                best_x = x
                report_best(best_val, best_x)

        return best_val, best_x