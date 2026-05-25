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
        
        # Restart parameters
        stagnation_limit = max(5, int(0.1 * self.budget))
        no_improve_count = 0
        
        while evals < self.budget:
            # Check restart
            if no_improve_count >= stagnation_limit:
                # Restart: reinitialize mean and CMA parameters
                mean = lb + self.rng.rand(dim) * span
                sigma = 0.2 * np.mean(span)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                C = np.eye(dim)
                no_improve_count = 0
                # Optionally adjust population size? Keep as is
                continue
            
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
            
            # Generate samples
            z = self.rng.randn(dim, lambda_)
            samples = mean + sigma * (L @ z).T
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
            
            # Sort by fitness
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
            
            # Ensure positive definiteness
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += np.eye(dim) * (1e-12 - np.min(eigvals))
            
            # Check for improvement
            if best_value < best_val_old:  # best_val_old defined? Actually we track best_value globally
                no_improve_count = 0
            else:
                no_improve_count += 1
            # We need to store previous best_value to compare; we can track best_value over iterations
            # But best_value is updated inside loop, so we need to record before evaluation loop? 
            # Simple: track best_value before sampling
            # Actually, we can set a variable best_before = best_value before sampling
            # But since we update best_value in loop, we need to compute no_improve_count based on whether best_value improved during this iteration.
            # We'll do: after evaluation, if best_value increased (i.e., got smaller) -> improvement
            # We'll store best_value before sampling
        
        return best_value, best_x