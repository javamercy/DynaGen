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
        x0 = self.rs.uniform(lb, ub, d)
        fbest = func(x0)
        xbest = x0.copy()
        report_best(fbest, xbest)
        total_evals = 1

        pop_multiplier = 1
        while total_evals < self.budget:
            lam = (8 + int(3 * np.log(d))) * pop_multiplier
            if total_evals + lam > self.budget:
                break
            xmean = self.rs.uniform(lb, ub, d)
            sigma = 0.5 * np.mean(ub - lb)
            C = np.eye(d) + 1e-12 * np.eye(d)
            mu = max(lam // 2, 1)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = (4.0 + mueff / d) / (d + 4.0 + 2.0 * mueff / d)
            cs = (mueff + 2.0) / (d + mueff + 5.0)
            c1 = 2.0 / ((d + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
            pc = np.zeros(d)
            ps = np.zeros(d)
            norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
            best_in_run = fbest
            no_improve_gens = 0
            max_no_improve = 100 + int(np.floor(d * 0.5))

            while total_evals + lam <= self.budget:
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-12 * np.eye(d)
                    B = np.linalg.cholesky(C)
                z = self.rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
                x = np.empty_like(x_raw)
                for i in range(lam):
                    xi = x_raw[i]
                    # mirror bounds
                    for j in range(d):
                        if xi[j] < lb[j]:
                            xi[j] = lb[j] + (lb[j] - xi[j])
                            if xi[j] > ub[j]:
                                xi[j] = ub[j] - (xi[j] - ub[j])
                        elif xi[j] > ub[j]:
                            xi[j] = ub[j] - (xi[j] - ub[j])
                            if xi[j] < lb[j]:
                                xi[j] = lb[j] + (lb[j] - xi[j])
                    x[i] = np.clip(xi, lb, ub)  # final safety clip
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
                    no_improve_gens = 0
                else:
                    no_improve_gens += 1
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
                # check condition number
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    cond = eigvals[-1] / max(eigvals[0], 1e-12)
                    if cond > 1e6:
                        break
                except np.linalg.LinAlgError:
                    pass
                if no_improve_gens >= max_no_improve:
                    break
            pop_multiplier *= 2
            if total_evals >= self.budget:
                break
        return fbest, xbest