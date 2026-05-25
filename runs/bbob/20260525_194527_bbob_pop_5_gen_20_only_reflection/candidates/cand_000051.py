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
        rs = self.rs
        budget = self.budget

        # initial point
        x = rs.uniform(lb, ub, d)
        fbest = func(x)
        xbest = x.copy()
        report_best(fbest, xbest)
        total_evals = 1

        popsize_multiplier = 1
        while total_evals < budget:
            lam = (8 + int(3 * np.log(d))) * popsize_multiplier
            if total_evals + lam > budget:
                break
            # restart: uniform initialization
            xmean = rs.uniform(lb, ub, d)
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

            while total_evals + lam <= budget:
                # generate points
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                z = rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
                # mirror out-of-bounds coordinates
                x = x_raw.copy()
                for i in range(lam):
                    xi = x[i]
                    mask_low = xi < lb
                    xi[mask_low] = lb[mask_low] + (lb[mask_low] - xi[mask_low])
                    mask_high = xi > ub
                    xi[mask_high] = ub[mask_high] - (xi[mask_high] - ub[mask_high])
                    xi = np.clip(xi, lb, ub)
                    x[i] = xi
                # evaluate
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
                    break

            # local refinement after restart, if budget remains
            if total_evals < budget:
                local_sigma = 0.1 * np.mean(ub - lb)
                local_lam = max(4, int(np.sqrt(d)))
                local_generations = 0
                max_local_generations = 5
                while (total_evals + local_lam <= budget) and (local_generations < max_local_generations):
                    z = rs.randn(local_lam, d)
                    x_try = xbest + local_sigma * z
                    for i in range(local_lam):
                        xi = x_try[i]
                        mask_low = xi < lb
                        xi[mask_low] = lb[mask_low] + (lb[mask_low] - xi[mask_low])
                        mask_high = xi > ub
                        xi[mask_high] = ub[mask_high] - (xi[mask_high] - ub[mask_high])
                        xi = np.clip(xi, lb, ub)
                        x_try[i] = xi
                    for i in range(local_lam):
                        fval = func(x_try[i])
                        total_evals += 1
                        if fval < fbest:
                            fbest = fval
                            xbest = x_try[i].copy()
                            report_best(fbest, xbest)
                    local_generations += 1

            popsize_multiplier *= 2
            if total_evals >= budget:
                break
        return fbest, xbest