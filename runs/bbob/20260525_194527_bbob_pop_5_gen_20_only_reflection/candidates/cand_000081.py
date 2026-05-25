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
        # Initial point
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
            # Restart: uniform initialization
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
                # Cholesky
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                z = self.rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
                # Mirror bounds
                x = np.empty_like(x_raw)
                for i in range(lam):
                    xi = x_raw[i]
                    # mirror until inside bounds, max 5 attempts
                    for _ in range(5):
                        out_low = xi < lb
                        out_high = xi > ub
                        if not np.any(out_low | out_high):
                            break
                        xi[out_low] = lb[out_low] + (lb[out_low] - xi[out_low])
                        xi[out_high] = ub[out_high] - (xi[out_high] - ub[out_high])
                    # if still out, clip
                    xi = np.clip(xi, lb, ub)
                    x[i] = xi
                # Evaluate
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
            # Local refinement after restart if budget remains
            if total_evals < self.budget:
                # run a few generations with small sigma around best
                lam_local = max(4 + int(3 * np.log(d)), 2)
                if total_evals + lam_local > self.budget:
                    break
                xmean = xbest.copy()
                sigma_local = sigma * 0.1
                C_local = np.eye(d) * (sigma_local**2) + 1e-9 * np.eye(d)
                # reset evolution path for local
                pc_local = np.zeros(d)
                ps_local = np.zeros(d)
                max_gen_local = 5
                for gen in range(max_gen_local):
                    if total_evals + lam_local > self.budget:
                        break
                    try:
                        B_local = np.linalg.cholesky(C_local)
                    except np.linalg.LinAlgError:
                        C_local += 1e-9 * np.eye(d)
                        B_local = np.linalg.cholesky(C_local)
                    z = self.rs.randn(lam_local, d)
                    x_raw = xmean + np.dot(z, B_local.T)
                    # mirror bounds
                    x_local = np.empty_like(x_raw)
                    for i in range(lam_local):
                        xi = x_raw[i]
                        for _ in range(5):
                            out_low = xi < lb
                            out_high = xi > ub
                            if not np.any(out_low | out_high):
                                break
                            xi[out_low] = lb[out_low] + (lb[out_low] - xi[out_low])
                            xi[out_high] = ub[out_high] - (xi[out_high] - ub[out_high])
                        xi = np.clip(xi, lb, ub)
                        x_local[i] = xi
                    fvals_local = np.empty(lam_local)
                    for i in range(lam_local):
                        fvals_local[i] = func(x_local[i])
                        total_evals += 1
                        if fvals_local[i] < fbest:
                            fbest = fvals_local[i]
                            xbest = x_local[i].copy()
                            report_best(fbest, xbest)
                    idx_local = np.argsort(fvals_local)
                    x_sorted_local = x_local[idx_local]
                    old_xmean_local = xmean.copy()
                    # use mu = min(lam_local//2, 1)
                    mu_local = max(lam_local // 2, 1)
                    weights_local = np.log(mu_local + 0.5) - np.log(np.arange(1, mu_local + 1))
                    weights_local /= weights_local.sum()
                    xmean = np.dot(weights_local, x_sorted_local[:mu_local])
                    # update C locally (simplified, no cumulation)
                    # just keep C_local fixed for simplicity
            popsize_multiplier *= 2
            if total_evals >= self.budget:
                break
        return fbest, xbest