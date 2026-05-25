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
        rng = self.rng
        budget = self.budget

        N = dim
        lambda_ = 4 + int(3 * np.log(N))
        mu_ = lambda_ // 2
        if mu_ < 1:
            mu_ = 1
        weights = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
        weights /= weights.sum()
        mueff = 1.0 / (weights ** 2).sum()
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        cs = (mueff + 2) / (N + mueff + 5)
        c1 = 2.0 / ((N + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((N + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

        mean = lb + rng.rand(dim) * (ub - lb)
        sigma = (ub - lb).mean() / 4.0
        C = np.eye(N)
        pc = np.zeros(N)
        ps = np.zeros(N)
        best_val = np.inf
        best_x = None
        evals = 0

        val = func(mean)
        evals += 1
        best_val = val
        best_x = mean.copy()
        report_best(best_val, best_x)

        while evals < budget:
            lambda_eff = min(lambda_, budget - evals)
            if lambda_eff < 1:
                break
            pop = []
            fitness = []
            for i in range(lambda_eff):
                z = rng.randn(N)
                y = np.linalg.cholesky(C) @ z
                x = mean + sigma * y
                x = np.clip(x, lb, ub)
                val = func(x)
                evals += 1
                fitness.append(val)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                if evals >= budget:
                    break
            if evals >= budget:
                break
            indices = np.argsort(fitness)
            pop = [pop[i] for i in indices]
            old_mean = mean.copy()
            mean = np.zeros(N)
            for i in range(mu_):
                mean += weights[i] * pop[i]
            y = (mean - old_mean) / sigma
            # Update evolution paths
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y
            # Compute invsqrtC
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            invsqrtC = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y)
            # Update covariance matrix
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
            for i in range(mu_):
                z = (pop[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            # Update step size
            chiN = np.sqrt(N) * (1 - 1.0 / (4 * N) + 1.0 / (21 * N ** 2))
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        return best_val, best_x