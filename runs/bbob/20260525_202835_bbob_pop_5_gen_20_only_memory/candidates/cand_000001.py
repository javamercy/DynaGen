import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initialize mean uniformly in bounds
        mean = np.random.uniform(lb, ub, dim)

        # Step size
        sigma = 0.2 * (ub - lb).mean()  # adaptive
        # Covariance matrix (identity)
        C = np.eye(dim)

        # Strategy parameters
        lambda_ = int(4 + 3 * np.log(dim))
        lambda_ = min(lambda_, budget)  # ensure at least one generation

        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        invsqrtC = np.eye(dim)
        eigeneval = 0
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial evaluation
        best_x = mean.copy()
        best_val = func(best_x)
        report_best(best_val, best_x)
        evals = 1

        # Main loop
        while evals < budget:
            # Sample population
            arz = np.random.randn(lambda_, dim)
            arx = mean + sigma * (arz @ (B * D).T)
            # Clip to bounds
            arx = np.clip(arx, lb, ub)
            # Evaluate
            arf = np.array([func(arx[i]) for i in range(lambda_)])
            evals += lambda_
            if evals > budget:
                # If we overrun, we already evaluated too many? But we break after loop
                break

            # Sort
            idx = np.argsort(arf)
            arx = arx[idx]
            arf = arf[idx]
            # Update best
            if arf[0] < best_val:
                best_val = arf[0]
                best_x = arx[0].copy()
                report_best(best_val, best_x)

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(weights, arx[:mu])

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * evals / lambda_)) / chiN < 1.4 + 2 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma

            # Update covariance matrix
            art = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * np.dot(weights * art.T, art)

            # Update step size
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

            # Update eigendecomposition every generation if needed
            if evals - eigeneval > lambda_ / (c1 + cmu) / dim / 10:
                eigeneval = evals
                C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.abs(D))
                invsqrtC = B @ np.diag(1.0 / D) @ B.T

            # Check budget left
            if evals >= budget:
                break

        return best_val, best_x