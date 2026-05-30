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
        # Initial random point
        mean = np.random.uniform(lb, ub)
        best_x = mean.copy()
        best_val = func(mean)
        evals = 1
        report_best(best_val, best_x)
        # CMA-ES parameters (explorative)
        sigma0 = 0.5 * np.mean(ub - lb)
        sigma = sigma0
        C = np.eye(dim)
        lam = 10 + int(5 * math.log(dim))
        lam = min(lam, self.budget - evals)
        if lam < 2:
            lam = max(2, self.budget - evals)
        mu = max(1, lam // 2)  # less selection pressure
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff) * 0.5  # reduced learning
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2) ** 2 + mueff)) * 0.5
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        last_improvement_evals = evals
        stagnation_limit = max(20, int(0.2 * self.budget))  # more patience
        restart_count = 0
        max_restarts = 10
        restart_seed = self.seed + 1

        while evals < self.budget:
            # Stagnation or periodic restart from random point
            if (evals - last_improvement_evals > stagnation_limit
                    and self.budget - evals > 5 and restart_count < max_restarts):
                np.random.seed(restart_seed)
                restart_seed += 1
                mean = np.random.uniform(lb, ub)
                sigma = sigma0
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
                lam = 10 + int(5 * math.log(dim))
                lam = min(lam, self.budget - evals)
                if lam < 2:
                    lam = max(2, self.budget - evals)
                mu = max(1, lam // 2)
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = weights / np.sum(weights)
                mueff = 1.0 / np.sum(weights ** 2)
                cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
                cs = (mueff + 2) / (dim + mueff + 5)
                c1 = 2.0 / ((dim + 1.3) ** 2 + mueff) * 0.5
                cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2) ** 2 + mueff)) * 0.5
                damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
                continue

            # Sample population
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                A = np.eye(dim)
            candidates = []
            for i in range(lam):
                z = np.random.randn(dim)
                x = mean + sigma * A @ z
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
                    last_improvement_evals = evals
            if len(vals) == 0:
                break
            idx = np.argsort(vals)
            candidates = [candidates[i] for i in idx]
            # Update mean
            old_mean = mean.copy()
            sorted_vals = sorted(vals)
            mean = np.sum([w * candidates[i] for i, w in enumerate(weights[:len(weights)])], axis=0)
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
                z = (candidates[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2
            # Update step size
            sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
            # Adjust lambda for remaining budget
            remaining = self.budget - evals
            if remaining < lam:
                lam = max(2, remaining)
        return best_val, best_x