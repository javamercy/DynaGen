import numpy as np

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
        x = self.rs.uniform(lb, ub, d)
        fbest = func(x)
        xbest = x.copy()
        report_best(fbest, xbest)
        total_evals = 1

        pop_mult = 1
        while total_evals < self.budget:
            lam = (8 + int(3 * np.log(d))) * pop_mult
            if total_evals + lam > self.budget:
                break
            # restart initialization: deterministic perturb-best
            if total_evals == 1:
                xmean = self.rs.uniform(lb, ub, d)
            else:
                # perturb best within 0.1 * range
                sigma_init = 0.1 * np.mean(ub - lb)
                xmean = xbest + sigma_init * self.rs.randn(d)
                xmean = np.clip(xmean, lb, ub)
            sigma = 0.5 * np.mean(ub - lb)
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            mu = max(lam // 2, 1)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = 4.0 / (d + 4.0)
            cs = (mueff + 2.0) / (d + mueff + 5.0)
            c1 = 2.0 / ((d + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
            norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
            best_in_run = fbest
            no_improve = 0
            max_no_improve = 100 + int(np.floor(d * 0.5))

            while total_evals + lam <= self.budget:
                # condition number check
                eigvals = np.linalg.eigvalsh(C)
                if np.max(eigvals) / np.min(eigvals) > 1e14:
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                z = self.rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
                x = np.empty_like(x_raw)
                for i in range(lam):
                    xi = x_raw[i]
                    attempts = 0
                    while np.any(xi < lb) or np.any(xi > ub):
                        attempts += 1
                        if attempts > 10:
                            # mirror
                            xi = np.where(xi < lb, lb + (lb - xi), xi)
                            xi = np.where(xi > ub, ub - (xi - ub), xi)
                            xi = np.clip(xi, lb, ub)
                            break
                        zi = self.rs.randn(d)
                        xi = xmean + sigma * np.dot(zi, B.T)
                    x[i] = xi
                fvals = np.empty(lam)
                for i in range(lam):
                    fvals[i] = func(x[i])
                    total_evals += 1
                    if fvals[i] < fbest:
                        fbest = fvals[i]
                        xbest = x[i].copy()
                        report_best(fbest, xbest)
                if np.min(fvals) < best_in_run:
                    best_in_run = np.min(fvals)
                    no_improve = 0
                else:
                    no_improve += 1
                idx = np.argsort(fvals)
                x_sorted = x[idx]
                old_xmean = xmean.copy()
                xmean = np.dot(weights, x_sorted[:mu])
                x_diff = (xmean - old_xmean) / sigma
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * x_diff
                try:
                    invB = np.linalg.solve(B, np.eye(d))
                except np.linalg.LinAlgError:
                    invB = np.linalg.inv(B)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invB, x_diff)
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                diff = x_sorted[:mu] - old_xmean
                diff_norm = diff / sigma
                C += cmu * np.dot(diff_norm.T, np.dot(np.diag(weights), diff_norm))
                C = (C + C.T) / 2.0
                ps_norm = np.linalg.norm(ps)
                sigma *= np.exp((cs / ds) * (ps_norm / norm_expected - 1.0))
                if no_improve >= max_no_improve:
                    break

            # local refinement after restart
            if total_evals + 5 <= self.budget:
                sigma_local = 0.01 * np.mean(ub - lb)
                for _ in range(5):
                    z = self.rs.randn(d)
                    xi = xbest + sigma_local * z
                    xi = np.clip(xi, lb, ub)
                    fval = func(xi)
                    total_evals += 1
                    if fval < fbest:
                        fbest = fval
                        xbest = xi.copy()
                        report_best(fbest, xbest)

            pop_mult *= 2
            if total_evals >= self.budget:
                break

        return fbest, xbest