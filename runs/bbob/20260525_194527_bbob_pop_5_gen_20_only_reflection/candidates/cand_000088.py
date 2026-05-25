import numpy as np
from math import log, exp, sqrt

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rs = np.random.RandomState(seed)

    def __call__(self, func):
        d = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        x0 = self.rs.uniform(lb, ub, d)
        fbest = func(x0)
        xbest = x0.copy()
        report_best(fbest, xbest)
        evals = 1

        multiplier = 1
        restart_count = 0
        while evals < self.budget:
            lam = (4 + int(3 * log(d))) * multiplier
            if evals + lam > self.budget:
                break
            
            # Initialize CMA-ES state
            if self.rs.rand() < 0.2 or fbest == np.inf:
                xmean = self.rs.uniform(lb, ub, d)
            else:
                xmean = xbest + self.rs.uniform(-0.1, 0.1, d) * (ub - lb)
                xmean = np.clip(xmean, lb, ub)
            sigma = 0.3 * np.mean(ub - lb)
            C = np.eye(d)
            mu = max(lam // 2, 1)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = 4.0 / (d + 4.0)
            cs = (mueff + 2.0) / (d + mueff + 5.0)
            c1 = 2.0 / ((d + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0/mueff) / ((d + 2.0)**2 + mueff))
            ds = 1.0 + 2.0 * max(0.0, sqrt((mueff - 1.0)/(d + 1.0)) - 1.0) + cs
            pc = np.zeros(d)
            ps = np.zeros(d)
            norm_expected = sqrt(d) * (1.0 - 1.0/(4.0*d) + 1.0/(21.0*d*d))
            best_in_run = fbest
            no_improve = 0
            max_no_improve = 50 + int(0.2 * d)
            cond_threshold = 1e14
            sigma_min = 1e-8 * np.mean(ub - lb)
            mean_bounds_range = np.mean(ub - lb)

            while evals + lam <= self.budget:
                # Condition number check
                try:
                    eigvals = np.linalg.eigh(C)[0]
                    cond = eigvals[-1] / max(eigvals[0], 1e-20)
                except:
                    cond = 1.0
                if cond > cond_threshold or sigma < sigma_min:
                    break
                # Cholesky decomposition
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                # Sample offspring
                z = self.rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
                # Resample for bounds
                x = np.empty_like(x_raw)
                for i in range(lam):
                    xi = x_raw[i]
                    attempts = 0
                    while np.any(xi < lb) or np.any(xi > ub):
                        attempts += 1
                        if attempts > 10:
                            xi = np.clip(xi, lb, ub)
                            break
                        zi = self.rs.randn(d)
                        xi = xmean + sigma * np.dot(zi, B.T)
                    x[i] = xi
                # Evaluate
                fvals = np.empty(lam)
                for i in range(lam):
                    fvals[i] = func(x[i])
                    evals += 1
                    if fvals[i] < fbest:
                        fbest = fvals[i]
                        xbest = x[i].copy()
                        report_best(fbest, xbest)
                # Stagnation check
                min_f = np.min(fvals)
                if min_f < best_in_run:
                    best_in_run = min_f
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= max_no_improve:
                    break
                # Selection and update
                idx = np.argsort(fvals)
                x_sorted = x[idx]
                old_xmean = xmean.copy()
                xmean = np.dot(weights, x_sorted[:mu])
                x_diff = (xmean - old_xmean) / sigma
                pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mueff) * x_diff
                # Update ps
                invB = np.linalg.solve(B, np.eye(d))
                ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * np.dot(invB, x_diff)
                # Update covariance
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                diff = x_sorted[:mu] - old_xmean
                diff_norm = diff / sigma
                C += cmu * np.dot(diff_norm.T, np.dot(np.diag(weights), diff_norm))
                C = (C + C.T) / 2.0
                ps_norm = np.linalg.norm(ps)
                sigma *= exp((cs/ds) * (ps_norm / norm_expected - 1.0))
                sigma = max(sigma, 1e-10 * np.mean(ub - lb))
            
            # Local refinement after restart
            remaining = self.budget - evals
            local_budget = min(int(0.1 * remaining), 50)
            if local_budget > 0:
                step = 0.01 * mean_bounds_range * (1 - evals / self.budget)
                for _ in range(local_budget):
                    # Random coordinate perturbation
                    coord = self.rs.randint(d)
                    for sign in [1, -1]:
                        if evals >= self.budget:
                            break
                        x_new = xbest.copy()
                        x_new[coord] += sign * step
                        x_new = np.clip(x_new, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if f_new < fbest:
                            fbest = f_new
                            xbest = x_new.copy()
                            report_best(fbest, xbest)
                            break  # accept first improvement
            
            # Increase multiplier
            if evals < 0.5 * self.budget:
                multiplier *= 2
            else:
                multiplier *= 1.5
            restart_count += 1
        return fbest, xbest