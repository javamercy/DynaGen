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
        # Initial point
        x = np.random.uniform(lb, ub)
        best_val = func(x)
        best_x = x.copy()
        evals = 1
        report_best(best_val, best_x)
        if self.budget <= 1:
            return best_val, best_x
        # Phase 1: CMA-ES (exploitation focused)
        mean = best_x.copy()
        sigma = 0.2 * np.mean(range_)
        C = np.eye(dim)
        lam = max(2, 10 + int(3 * math.log(dim)))
        if lam > self.budget - evals:
            lam = max(2, self.budget - evals)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff) * 2.0
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)) * 2.0
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        phase1_budget = int(0.7 * self.budget)
        while evals < min(phase1_budget, self.budget):
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
            remaining = self.budget - evals
            if remaining < lam:
                lam = max(2, remaining)
            if sigma < 1e-10 * np.mean(range_):
                sigma = 0.01 * np.mean(range_)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
        # Phase 2: Pattern search around best
        step = 0.05 * np.mean(range_)
        shrink = 0.5
        while evals < self.budget:
            improved = False
            # Coordinate-wise pattern
            for d in range(dim):
                if evals >= self.budget:
                    break
                # Positive direction
                new_x = best_x.copy()
                new_x[d] = np.clip(best_x[d] + step, lb[d], ub[d])
                if not np.allclose(new_x, best_x):
                    val = func(new_x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = new_x
                        report_best(best_val, best_x)
                        improved = True
                        break  # Restart pattern from new best
                # Negative direction
                new_x = best_x.copy()
                new_x[d] = np.clip(best_x[d] - step, lb[d], ub[d])
                if not np.allclose(new_x, best_x):
                    val = func(new_x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = new_x
                        report_best(best_val, best_x)
                        improved = True
                        break
            if not improved:
                step *= shrink
                if step < 1e-12 * np.mean(range_):
                    break
        return best_val, best_x