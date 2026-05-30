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

        # Small initial Latin hypercube sample
        n_initial = max(2, min(5, dim * 2))
        def lhs_sample(n, d, lb, ub):
            samples = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                samples[:, i] = (perm + np.random.uniform(size=n)) / n * (ub[i] - lb[i]) + lb[i]
            return samples
        initial_pop = lhs_sample(n_initial, dim, lb, ub)
        for x in initial_pop:
            val = func(x)
            total_evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # CMA-ES parameters for exploitation
        lam = max(2, int(4 + 3 * math.log(dim)))
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

        mean = best_x.copy() if best_x is not None else np.random.uniform(lb, ub)
        sigma = 0.2 * np.mean(range_)  # smaller initial step
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Single CMA-ES run
        best_val_in_run = best_val
        no_improve_count = 0
        max_no_improve = 5

        while total_evals < self.budget:
            # Stagnation check
            if best_val_in_run == best_val:
                no_improve_count += 1
            else:
                no_improve_count = 0
                best_val_in_run = best_val

            if no_improve_count >= max_no_improve or sigma < 1e-6 * np.mean(range_):
                break

            # Adjust lambda based on remaining budget
            remaining = self.budget - total_evals
            if remaining < lam:
                lam = max(2, remaining)
                mu = lam // 2
                if mu < 1:
                    mu = 1
                if lam >= 2:
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights = weights / np.sum(weights)
            if lam < 2:
                break

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

            if len(vals) == 0:
                break

            idx = np.argsort(vals)
            candidates = [candidates[i] for i in idx]
            x_old = mean.copy()
            mean = np.zeros(dim)
            for i in range(mu):
                mean += weights[i] * candidates[i]
            np.clip(mean, lb, ub, out=mean)

            z_mean = (mean - x_old) / sigma
            try:
                inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                inv_sqrt_C = np.eye(dim)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C @ z_mean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**(2*total_evals/lam)) < (1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean

            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                z = (candidates[i] - x_old) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2

            sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))

        # Local search after CMA-ES
        if total_evals < self.budget:
            local_sigma = 0.1 * np.mean(range_)
            while total_evals < self.budget:
                candidate = best_x + local_sigma * np.random.randn(dim)
                np.clip(candidate, lb, ub, out=candidate)
                val = func(candidate)
                total_evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                    local_sigma = 0.1 * np.mean(range_)  # reset on improvement
                else:
                    local_sigma *= 0.9
                if local_sigma < 1e-12:
                    break

        return best_val, best_x