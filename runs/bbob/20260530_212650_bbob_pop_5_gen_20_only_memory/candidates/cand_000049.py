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
        if self.budget == 1:
            return best_val, best_x
        # Phase 1: CMA-ES with aggressive exploitation
        mean = best_x.copy()
        sigma = 0.2 * np.mean(range_)
        C = np.eye(dim)
        lam = max(2, 10 + int(3 * math.log(dim)))
        if lam > self.budget - evals:
            lam = max(2, self.budget - evals)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 * 2 / ((dim + 1.3)**2 + mueff)
        cmu = 2 * min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        last_improve = evals
        stagnation_limit = max(10, int(0.1 * self.budget))
        switch_to_local = False
        # CMA-ES loop
        while evals < self.budget and not switch_to_local:
            # Adjust lam if necessary
            remaining = self.budget - evals
            if remaining < lam:
                lam = max(2, remaining)
                mu = lam // 2
                if mu > 0:
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= weights.sum()
                    mueff = 1.0 / np.sum(weights**2)
                    cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
                    cs = (mueff + 2) / (dim + mueff + 5)
                    c1 = 4 / ((dim + 1.3)**2 + mueff)
                    cmu = 2 * min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
                    damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
            # Sample population
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                A = np.eye(dim)
            candidates = []
            for _ in range(lam):
                z = np.random.randn(dim)
                cand = mean + sigma * A @ z
                cand = np.clip(cand, lb, ub)
                candidates.append(cand)
            # Evaluate
            vals = []
            for cand in candidates:
                if evals >= self.budget:
                    break
                val = func(cand)
                vals.append(val)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = cand.copy()
                    report_best(best_val, best_x)
                    last_improve = evals
            if len(vals) == 0:
                break
            # Update mean and covariance
            idx = np.argsort(vals)
            candidates = [candidates[i] for i in idx]
            old_mean = mean.copy()
            mean = np.sum([weights[i] * candidates[i] for i in range(mu)], axis=0)
            mean = np.clip(mean, lb, ub)
            z_mean = (mean - old_mean) / sigma
            try:
                invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                invsqrtC = np.eye(dim)
            ps = (1 - cs) * ps + math.sqrt(cs*(2-cs)*mueff) * invsqrtC @ z_mean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1-cs)**(2*evals/lam)) < (1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc*(2-cc)*mueff) * z_mean
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
            for i in range(mu):
                z = (candidates[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2
            sigma *= math.exp((cs/damps) * (np.linalg.norm(ps)/math.sqrt(dim) - 1))
            if sigma < 1e-10 * np.mean(range_):
                sigma = 0.01 * np.mean(range_)
                C = np.eye(dim)
                pc.fill(0)
                ps.fill(0)
            # Check stagnation
            if evals - last_improve > stagnation_limit:
                switch_to_local = True
        # Phase 2: Pattern search with random leaps
        step = 0.05 * np.mean(range_)
        min_step = 1e-12 * np.mean(range_)
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
                        break
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
                # Random leap
                if step > min_step:
                    new_x = best_x + np.random.randn(dim) * step
                    new_x = np.clip(new_x, lb, ub)
                    if evals < self.budget:
                        val = func(new_x)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = new_x
                            report_best(best_val, best_x)
                            improved = True
                if not improved:
                    step *= shrink
                    if step < min_step:
                        break
        return best_val, best_x