import numpy as np
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        range_ = ub - lb
        best_val = float('inf')
        best_x = None
        total_evals = 0

        # Latin Hypercube sampling for initial population
        def lhs_sample(n, d, lb, ub):
            samples = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                samples[:, i] = (perm + np.random.uniform(size=n)) / n * (ub[i] - lb[i]) + lb[i]
            return samples

        # Initial population
        init_n = max(2, min(20, dim*5))
        initial_pop = lhs_sample(init_n, dim, lb, ub)
        for x in initial_pop:
            val = func(x)
            total_evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Main loop
        while total_evals < self.budget:
            remaining = self.budget - total_evals
            if remaining <= 1:
                break
            # Set CMA-ES parameters
            lam = max(2, int(8 + 3 * math.log(dim)))
            lam = min(lam, remaining)
            if lam < 2:
                lam = remaining
            mu = lam // 2
            if mu < 1:
                mu = 1
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
            damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs

            # Initialize mean and step size
            if best_x is None:
                mean = np.random.uniform(lb, ub)
            else:
                # Random convex combination of best and random point for diversity
                alpha = np.random.uniform(0.0, 0.5)
                rand_point = np.random.uniform(lb, ub)
                mean = alpha * best_x + (1 - alpha) * rand_point
                np.clip(mean, lb, ub, out=mean)
            sigma = 0.6 * np.mean(range_)  # larger initial step
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            best_val_in_run = best_val
            no_improve_count = 0
            max_no_improve = max(10, int(remaining * 0.1))

            while total_evals < self.budget:
                # Check stagnation
                if best_val_in_run == best_val:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                    best_val_in_run = best_val

                # Restart conditions: stagnation, small sigma, or random (5% per generation)
                rand_restart = np.random.uniform() < 0.05
                if no_improve_count >= max_no_improve or sigma < 1e-8 * np.mean(range_) or rand_restart:
                    break

                # Random perturbation of mean (2% per generation) for exploration
                if np.random.uniform() < 0.02:
                    perturbation = np.random.randn(dim) * sigma
                    mean = mean + perturbation
                    np.clip(mean, lb, ub, out=mean)

                # Sample offspring
                try:
                    A = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    A = np.eye(dim)
                candidates = []
                for i in range(lam):
                    z = np.random.randn(dim)
                    x = mean + sigma * A @ z
                    np.clip(x, lb, ub, out=x)
                    candidates.append(x)

                # Evaluate offspring
                vals = []
                for x in candidates:
                    if total_evals >= self.budget:
                        break
                    val = func(x)
                    total_evals += 1
                    vals.append(val)
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

                if len(vals) < 1:
                    break

                # Sort
                idx = np.argsort(vals)[:mu]
                candidates = [candidates[i] for i in idx]
                x_old = mean.copy()
                # Update mean
                mean = np.zeros(dim)
                for i in range(mu):
                    mean += weights[i] * candidates[i]
                np.clip(mean, lb, ub, out=mean)

                # Update evolution paths
                z_mean = (mean - x_old) / sigma
                try:
                    inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    inv_sqrt_C = np.eye(dim)
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C @ z_mean
                hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**(2*total_evals/lam)) < (1.4 + 2/(dim+1))
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean

                # Update covariance
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                for i in range(mu):
                    z = (candidates[i] - x_old) / sigma
                    C += cmu * weights[i] * np.outer(z, z)
                C = (C + C.T) / 2

                # Update step size
                sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))

                # Adjust lambda for remaining budget
                remaining = self.budget - total_evals
                if remaining < lam:
                    lam = max(2, remaining)
                    mu = lam // 2
                    if mu < 1:
                        mu = 1
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights = weights / np.sum(weights)

        return best_val, best_x