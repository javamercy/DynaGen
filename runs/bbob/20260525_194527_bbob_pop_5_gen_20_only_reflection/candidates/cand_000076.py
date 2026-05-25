import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rs = np.random.RandomState(seed)

    def reflect(self, x, lb, ub):
        y = x.copy()
        for i in range(len(y)):
            while y[i] < lb[i] or y[i] > ub[i]:
                if y[i] < lb[i]:
                    y[i] = lb[i] + (lb[i] - y[i])
                if y[i] > ub[i]:
                    y[i] = ub[i] - (y[i] - ub[i])
        return y

    def __call__(self, func):
        d = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        x = self.rs.uniform(lb, ub, d)
        fbest = func(x)
        xbest = x.copy()
        report_best(fbest, xbest)
        total_evals = 1

        popsize_multiplier = 1
        while total_evals < self.budget:
            lam = (8 + int(3 * np.log(d))) * popsize_multiplier
            if total_evals + lam > self.budget:
                break
            xmean = self.rs.uniform(lb, ub, d)
            sigma = 0.5 * np.mean(ub - lb)
            C = np.eye(d) + 1e-9 * np.eye(d)
            mu = max(lam // 2, 1)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = 4.0 / (d + 4.0)
            cs = (mueff + 2.0) / (d + mueff + 5.0)
            c1 = 2.0 / ((d + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
            pc = np.zeros(d)
            ps = np.zeros(d)
            norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
            best_in_run = fbest
            no_improve_generations = 0
            max_no_improve = 100 + int(np.floor(d * 0.5))

            while total_evals + lam <= self.budget:
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                z = self.rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
                x = np.array([self.reflect(xi, lb, ub) for xi in x_raw])
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
                    no_improve_generations = 0
                else:
                    no_improve_generations += 1
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
                if no_improve_generations >= max_no_improve:
                    if total_evals < self.budget:
                        lam_local = max(4 + int(3 * np.log(d)), 1)
                        if total_evals + lam_local <= self.budget:
                            try:
                                B_local = np.linalg.cholesky(C)
                            except np.linalg.LinAlgError:
                                C += 1e-9 * np.eye(d)
                                B_local = np.linalg.cholesky(C)
                            z_local = self.rs.randn(lam_local, d)
                            x_raw_local = xbest + 0.1 * sigma * np.dot(z_local, B_local.T)
                            x_local = np.array([self.reflect(xi, lb, ub) for xi in x_raw_local])
                            for i in range(lam_local):
                                f = func(x_local[i])
                                total_evals += 1
                                if f < fbest:
                                    fbest = f
                                    xbest = x_local[i].copy()
                                    report_best(fbest, xbest)
                    break
            popsize_multiplier *= 2
            if total_evals >= self.budget:
                break
        return fbest, xbest