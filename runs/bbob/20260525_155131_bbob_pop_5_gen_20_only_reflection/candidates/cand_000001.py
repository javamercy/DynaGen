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
        span = ub - lb
        
        # Initial mean
        mean = lb + self.rng.rand(dim) * span
        
        # Evaluate initial mean
        best_value = func(mean)
        best_x = mean.copy()
        evals = 1
        report_best(best_value, best_x)
        
        # Population size
        lambda_ = max(4, min(self.budget // 10, 4 + int(3 * np.log(dim))))
        if lambda_ < 1:
            lambda_ = 1
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control
        cs = (mueff + 2) / (dim + mueff + 3)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        ps = np.zeros(dim)
        
        # Covariance adaptation
        ccov = (1.0 / mueff) * 2.0 / ((dim + 1.3)**2) + 1.0 - 1.0 / mueff
        pc = np.zeros(dim)
        C = np.eye(dim)
        sigma = 0.2 * np.mean(span)
        
        # Main loop
        while evals < self.budget:
            # Sample population
            remaining = self.budget - evals
            if remaining < lambda_:
                lambda_ = remaining
                if lambda_ < 1:
                    break
                mu = lambda_ // 2
                if mu < 1:
                    mu = 1
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = weights / np.sum(weights)
                mueff = 1.0 / np.sum(weights**2)
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            
            samples = mean + sigma * (L @ self.rng.randn(dim, lambda_)).T
            # Clip to bounds
            samples = np.clip(samples, lb, ub)
            
            # Evaluate
            vals = np.full(lambda_, np.inf)
            for i in range(lambda_):
                if evals >= self.budget:
                    break
                vals[i] = func(samples[i])
                evals += 1
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = samples[i].copy()
                    report_best(best_value, best_x)
            
            # Sort
            idx = np.argsort(vals)
            samples_sorted = samples[idx]
            
            # Update mean
            mean_old = mean.copy()
            mean = np.dot(weights, samples_sorted[:mu])
            
            # Update step size path
            invsqrtC = np.linalg.inv(L)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - mean_old)) / sigma
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / (np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))) - 1))
            
            # Update covariance path
            pc = (1 - ccov) * pc + np.sqrt(ccov * (2 - ccov) * mueff) * (mean - mean_old) / sigma
            C = (1 - ccov) * C + ccov * np.outer(pc, pc)
            # Ensure symmetry
            C = (C + C.T) / 2
            # Ensure positive definiteness with small eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += np.eye(dim) * (1e-12 - np.min(eigvals))
            
        return best_value, best_x