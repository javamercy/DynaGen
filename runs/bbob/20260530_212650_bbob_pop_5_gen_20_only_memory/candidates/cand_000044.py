import math
import numpy as np

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
        # Initialization
        mean = np.random.uniform(lb, ub)
        best_x = mean.copy()
        best_val = func(mean)
        evals = 1
        report_best(best_val, best_x)
        # CMA-ES parameters (exploration-focused)
        sigma = 0.8 * np.mean(ub - lb)  # larger initial step size
        C = np.eye(dim)
        lam = 15 + int(5 * math.log(dim))  # larger population
        lam = min(lam, self.budget - evals)
        if lam < 2:
            lam = max(2, self.budget - evals)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        # Stagnation tracking
        last_improvement_evals = evals
        stagnation_limit = max(10, int(0.1 * self.budget))  # smaller threshold
        restart_count = 0
        max_restarts = 3
        random_fraction = 0.2  # fraction of random immigrants each generation
        # Main loop
        while evals < self.budget:
            # Adjust lam based on remaining budget
            remaining = self.budget - evals
            if remaining < lam:
                lam = max(2, remaining)
                mu = lam // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = weights / np.sum(weights)
                mueff = 1.0 / np.sum(weights ** 2)
                cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
                cs = (mueff + 2) / (dim + mueff + 5)
                c1 = 2 / ((dim + 1.3) ** 2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2) ** 2 + mueff))
                damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
            # Check stagnation and restart if needed
            if (evals - last_improvement_evals > stagnation_limit and 
                remaining > 5 and restart_count < max_restarts):
                # Restart with random mean
                mean = np.random.uniform(lb, ub)
                sigma = 0.8 * np.mean(ub - lb)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                # Evaluate new mean
                if evals < self.budget:
                    val = func(mean)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = mean.copy()
                        report_best(best_val, best_x)
                    last_improvement_evals = evals
                restart_count += 1
                # Recompute lam and parameters for new budget
                lam = 15 + int(5 * math.log(dim))
                lam = min(lam, self.budget - evals)
                if lam < 2:
                    lam = max(2, self.budget - evals)
                mu = lam // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = weights / np.sum(weights)
                mueff = 1.0 / np.sum(weights ** 2)
                cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
                cs = (mueff + 2) / (dim + mueff + 5)
                c1 = 2 / ((dim + 1.3) ** 2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2) ** 2 + mueff))
                damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
            # Sample population: some from CMA, some random immigrants
            n_cma = lam - int(random_fraction * lam)
            n_random = lam - n_cma
            candidates = []
            # CMA samples
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                A = np.eye(dim)
            for i in range(n_cma):
                z = np.random.randn(dim)
                x = mean + sigma * A @ z
                x = np.clip(x, lb, ub)
                candidates.append(x)
            # Random immigrants
            for i in range(n_random):
                x = np.random.uniform(lb, ub)
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
                    last_improvement_evals = evals
            if len(vals) == 0:
                break
            # Sort and select top mu
            idx = np.argsort(vals)[:mu]
            # Update mean
            old_mean = mean.copy()
            mean = np.sum([weights[i] * candidates[idx[i]] for i in range(mu)], axis=0)
            mean = np.clip(mean, lb, ub)
            # Update evolution paths
            z_mean = (mean - old_mean) / sigma
            try:
                invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
            except:
                invsqrtC = np.eye(dim)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ z_mean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2*evals/lam)) < (1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean
            # Update covariance
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                z = (candidates[idx[i]] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2
            # Update step size
            sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
            # Adjust lam for next iteration (already at top of loop, but keep here for clarity)
        return best_val, best_x