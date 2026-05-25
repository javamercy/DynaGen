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
        
        best_value = np.inf
        best_x = np.zeros(dim)
        
        # Initial mean
        mean = lb + self.rng.rand(dim) * span
        best_value = func(mean)
        best_x = mean.copy()
        report_best(best_value, best_x)
        evals = 1
        
        # Base population size
        base_lambda = max(4, min(self.budget // 10, 4 + int(3 * np.log(dim))))
        if base_lambda < 1:
            base_lambda = 1
        
        # Restart parameters
        stagnation_limit = max(1, self.budget // 10)
        
        # Main loop with restarts
        restart = 0
        while evals < self.budget:
            # Increase population size at each restart
            lambda_ = base_lambda * (2 ** restart)
            if lambda_ > self.budget - evals:
                lambda_ = self.budget - evals
            if lambda_ < 1:
                break
            
            mu = lambda_ // 2
            if mu < 1:
                mu = 1
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            cs = (mueff + 2) / (dim + mueff + 3)
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            ps = np.zeros(dim)
            
            ccov = (1.0 / mueff) * 2.0 / ((dim + 1.3)**2) + 1.0 - 1.0 / mueff
            pc = np.zeros(dim)
            C = np.eye(dim)
            sigma = 0.2 * np.mean(span)
            
            # Stagnation tracking
            last_improve = evals
            
            # Inner CMA-ES run
            while evals < self.budget:
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
                
                # Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                # Generate samples with mirrored pairs
                half = lambda_ // 2
                remainder = lambda_ % 2
                z = self.rng.randn(dim, half)
                samples = np.zeros((lambda_, dim))
                for i in range(half):
                    samples[2*i] = mean + sigma * (L @ z[:, i])
                    samples[2*i+1] = mean - sigma * (L @ z[:, i])
                if remainder == 1:
                    samples[-1] = mean + sigma * (L @ self.rng.randn(dim))
                
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
                        last_improve = evals
                
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
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += np.eye(dim) * (1e-12 - np.min(eigvals))
                
                # Stagnation check
                if evals - last_improve > stagnation_limit:
                    break
            
            restart += 1
        
        return best_value, best_x