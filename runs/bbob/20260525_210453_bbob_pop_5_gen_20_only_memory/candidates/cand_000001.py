import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.best_value = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initialize mean at center
        mean = (lb + ub) / 2.0
        sigma = (ub - lb).max() / 4.0
        if sigma == 0:
            sigma = 1.0

        # Covariance matrix: start with identity
        C = np.eye(dim)
        # Evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Hyperparameters (standard CMA-ES defaults)
        lambda_ = int(4 + 3 * np.log(dim))
        lambda_ = min(lambda_, budget // 2)
        if lambda_ < 3:
            lambda_ = min(budget, 3)
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        # Weights: logarithmic
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

        # Initial evaluation of mean
        x0 = np.clip(mean, lb, ub)
        y0 = func(x0)
        calls = 1
        self.best_value = y0
        self.best_x = x0.copy()
        # report_best is available globally
        try:
            report_best(self.best_value, self.best_x)
        except NameError:
            pass

        generation = 0
        while calls < budget:
            generation += 1
            # Sample population
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            arz = self.rng.randn(lambda_, dim)
            arx = mean + sigma * np.dot(arz, B.T)

            # Clip to bounds and evaluate
            arx_clip = np.clip(arx, lb, ub)
            ary = np.array([func(arx_clip[i]) for i in range(lambda_)])
            calls += lambda_
            # If we exceed budget, truncate earlier evaluations
            if calls > budget:
                # Already evaluated, but we must not exceed budget
                # We'll just break and not use last evaluations if they exceed
                # Actually we counted calls, but we should ensure we don't use more than budget
                # Since we already called func lambda_ times, we break after reporting best
                pass

            # Sort by fitness
            indices = np.argsort(ary)
            ary_sorted = ary[indices]
            arx_sorted = arx_clip[indices]
            arz_sorted = arz[indices]

            # Update best
            if ary_sorted[0] < self.best_value:
                self.best_value = ary_sorted[0]
                self.best_x = arx_sorted[0].copy()
                try:
                    report_best(self.best_value, self.best_x)
                except NameError:
                    pass

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(weights, arx_sorted[:mu])

            # Update evolution paths
            zmean = np.dot(weights, arz_sorted[:mu])
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * np.dot(B, zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * calls / lambda_)) < (1.4 + 2.0 / (dim + 1.0)))
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * np.dot(B, zmean)

            # Update covariance matrix
            artmp = arx_sorted[:mu] - old_mean
            C = (1.0 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * np.dot((weights * artmp.T), artmp) / (sigma ** 2)

            # Update step size
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(dim) - 1.0))

            # Enforce symmetry and positive definiteness
            C = np.triu(C) + np.triu(C, 1).T
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-12 * np.eye(dim)

            # If we exceeded budget, stop
            if calls >= budget:
                break

        # Final return
        return self.best_value, self.best_x