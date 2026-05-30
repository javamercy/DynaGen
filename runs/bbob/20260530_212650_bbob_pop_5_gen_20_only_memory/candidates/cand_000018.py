import numpy as np
import math

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial point
        mean = np.random.uniform(lb, ub)
        best_x = mean.copy()
        best_val = func(mean)
        evals = 1
        report_best(best_val, best_x)

        # ES parameters
        lambda_ = 4 + int(3 * math.log(dim))
        lambda_ = min(lambda_, self.budget - evals)
        if lambda_ < 2:
            lambda_ = max(2, self.budget - evals)
        mu = max(1, lambda_ // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        cs = (mueff + 2) / (dim + mueff + 5)
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
        ps = np.zeros(dim)
        sigma = 0.2 * np.mean(ub - lb)
        last_improvement = evals
        stagnation_limit = max(10, int(0.1 * self.budget))
        restart_count = 0
        max_restarts = 5

        while evals < self.budget:
            # Check stagnation
            if (evals - last_improvement > stagnation_limit and
                self.budget - evals > 5 and restart_count < max_restarts):
                mean = best_x.copy()
                sigma = 0.2 * np.mean(ub - lb)
                ps = np.zeros(dim)
                if evals < self.budget:
                    val = func(mean)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = mean.copy()
                        report_best(best_val, best_x)
                        last_improvement = evals
                restart_count += 1
                # Recompute lambda for remaining budget
                lambda_ = 4 + int(3 * math.log(dim))
                lambda_ = min(lambda_, self.budget - evals)
                if lambda_ < 2:
                    lambda_ = max(2, self.budget - evals)
                mu = max(1, lambda_ // 2)
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = weights / np.sum(weights)
                mueff = 1.0 / np.sum(weights ** 2)
                cs = (mueff + 2) / (dim + mueff + 5)
                damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
                continue

            # Sample offspring
            candidates = []
            for _ in range(lambda_):
                z = np.random.randn(dim)
                x = mean + sigma * z
                x = np.clip(x, lb, ub)
                candidates.append(x)
            # Evaluate
            vals = []
            for x in candidates:
                if evals >= self.budget:
                    break
                val = func(x)
                vals.append(val)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                    last_improvement = evals
            if len(vals) == 0:
                break
            # Sort
            idx = np.argsort(vals)
            candidates = [candidates[i] for i in idx]
            # Update mean (truncation)
            old_mean = mean.copy()
            mean = np.sum([w * candidates[i] for i, w in enumerate(weights[:mu])], axis=0)
            mean = np.clip(mean, lb, ub)
            # Update path
            z_mean = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * z_mean
            # Update sigma
            sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
            sigma = max(sigma, 1e-12 * np.mean(ub - lb))
            # Adjust lambda for remaining budget
            remaining = self.budget - evals
            if remaining < lambda_:
                lambda_ = max(2, remaining)
                mu = max(1, lambda_ // 2)
                if mu > lambda_:
                    mu = lambda_
                if mu > 0:
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights = weights / np.sum(weights)
                    mueff = 1.0 / np.sum(weights ** 2)
                    cs = (mueff + 2) / (dim + mueff + 5)
                    damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs

        return best_val, best_x