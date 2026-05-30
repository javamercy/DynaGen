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
        # CMA-ES initialization
        mean = np.random.uniform(lb, ub)
        best_x = mean.copy()
        best_val = func(mean)
        evals = 1
        report_best(best_val, best_x)
        sigma = 0.5 * np.mean(ub - lb)
        C = np.eye(dim)
        lam = 4 + int(3 * math.log(dim))
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
        # CMA-ES phase until 80% budget used
        cma_budget = int(0.8 * self.budget)
        while evals < cma_budget:
            remaining = cma_budget - evals
            if remaining < lam:
                lam = max(2, remaining)
                if lam < 2:
                    break
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
                if evals >= cma_budget:
                    break
                val = func(x)
                vals.append(val)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            if len(vals) == 0:
                break
            idx = np.argsort(vals)
            candidates = [candidates[i] for i in idx]
            old_mean = mean.copy()
            mean = np.sum([w * candidates[i] for i, w in enumerate(weights[:len(weights)])], axis=0)
            mean = np.clip(mean, lb, ub)
            z_mean = (mean - old_mean) / sigma
            try:
                invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
            except:
                invsqrtC = np.eye(dim)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ z_mean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2*evals/lam)) < (1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                z = (candidates[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2
            sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
        # Coordinate descent phase on best_x for remaining budget
        step_sizes = (ub - lb) * 0.1
        while evals < self.budget:
            # randomly choose dimension
            d = np.random.randint(dim)
            step = step_sizes[d]
            # positive direction
            x_candidate = best_x.copy()
            x_candidate[d] = np.clip(best_x[d] + step, lb[d], ub[d])
            val_candidate = func(x_candidate)
            evals += 1
            if val_candidate < best_val:
                best_val = val_candidate
                best_x = x_candidate.copy()
                report_best(best_val, best_x)
                step_sizes[d] *= 1.2
            else:
                # negative direction
                x_candidate[d] = np.clip(best_x[d] - step, lb[d], ub[d])
                val_candidate = func(x_candidate)
                evals += 1
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_x = x_candidate.copy()
                    report_best(best_val, best_x)
                    step_sizes[d] *= 1.2
                else:
                    step_sizes[d] *= 0.5
            step_sizes[d] = max(step_sizes[d], 1e-10)
            if evals >= self.budget:
                break
        return best_val, best_x