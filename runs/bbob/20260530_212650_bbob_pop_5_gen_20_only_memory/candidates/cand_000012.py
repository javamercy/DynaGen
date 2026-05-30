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
        range_len = np.mean(ub - lb)
        # Initialization
        mean = np.random.uniform(lb, ub)
        best_x = mean.copy()
        best_val = func(mean)
        evals = 1
        report_best(best_val, best_x)
        # CMA-ES parameters (more exploitative)
        sigma = 0.2 * range_len
        C = np.eye(dim)
        lam = 4 + int(2 * math.log(dim + 1))
        lam = min(lam, max(2, self.budget - evals))
        mu = max(1, lam // 3)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        # Stagnation tracking
        last_improvement_evals = evals
        stagnation_limit = max(5, int(0.1 * self.budget))
        restart_count = 0
        max_restarts = 4
        sigma_restart_factor = 1.0
        # Main loop
        while evals < self.budget:
            # Check stagnation and restart
            if (evals - last_improvement_evals > stagnation_limit and
                self.budget - evals > 5 and restart_count < max_restarts):
                mean = np.random.uniform(lb, ub)
                sigma_restart_factor *= 0.5
                sigma = 0.2 * range_len * sigma_restart_factor
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                if evals < self.budget:
                    val = func(mean)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = mean.copy()
                        report_best(best_val, best_x)
                    last_improvement_evals = evals
                restart_count += 1
                lam = 4 + int(2 * math.log(dim + 1))
                lam = min(lam, max(2, self.budget - evals))
                mu = max(1, lam // 3)
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = weights / np.sum(weights)
                mueff = 1.0 / np.sum(weights ** 2)
            # Sample population
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                A = np.eye(dim)
            candidates = []
            for i in range(lam):
                if evals >= self.budget:
                    break
                z = np.random.randn(dim)
                x = mean + sigma * A @ z
                x = np.clip(x, lb, ub)
                candidates.append(x)
            if len(candidates) == 0:
                break
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
            # Sort
            idx = np.argsort(vals)
            candidates = [candidates[i] for i in idx]
            # Update mean
            old_mean = mean.copy()
            mean = np.sum([w * candidates[i] for i, w in enumerate(weights[:len(weights)])], axis=0)
            mean = np.clip(mean, lb, ub)
            # Update evolution paths
            z_mean = (mean - old_mean) / sigma
            try:
                invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
            except:
                invsqrtC = np.eye(dim)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ z_mean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * evals / lam)) < (1.4 + 2 / (dim + 1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean
            # Update covariance
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(min(mu, len(candidates))):
                z = (candidates[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2
            # Update step size
            sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
            # Adjust lambda for remaining budget
            remaining = self.budget - evals
            if remaining < lam:
                lam = max(2, remaining)
        # Final local refinement if budget remains
        if self.budget - evals >= 3:
            # Use a small population around best
            local_sigma = 0.01 * range_len
            local_lam = min(3, self.budget - evals)
            for _ in range(3):  # up to 3 generations
                if self.budget - evals < local_lam:
                    break
                for i in range(local_lam):
                    if evals >= self.budget:
                        break
                    x = best_x + local_sigma * np.random.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
        return best_val, best_x