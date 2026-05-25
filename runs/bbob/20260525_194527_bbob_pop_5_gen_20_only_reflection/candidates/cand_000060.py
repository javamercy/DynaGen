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

        popsize_multiplier = 1
        while total_evals < self.budget:
            lam = (8 + int(3 * np.log(d))) * popsize_multiplier
            if total_evals + lam > self.budget:
                break
            # Uniform reinitialization
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
                # Mirroring for bounds
                x = x_raw.copy()
                for i in range(lam):
                    for j in range(d):
                        if x[i, j] < lb[j]:
                            x[i, j] = lb[j] + (lb[j] - x[i, j])
                        if x[i, j] > ub[j]:
                            x[i, j] = ub[j] - (x[i, j] - ub[j])
                        # Final clip just in case
                        if x[i, j] < lb[j] or x[i, j] > ub[j]:
                            x[i, j] = np.clip(x[i, j], lb[j], ub[j])
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
            # Post-stagnation local refinement if budget allows
            if no_improve_generations >= max_no_improve and total_evals < self.budget:
                local_lam = max(4, int(np.floor(d * 0.5)))
                remaining = self.budget - total_evals
                if remaining >= local_lam:
                    local_gen_max = max(1, int(np.floor(remaining / local_lam)))
                    local_gen_max = min(local_gen_max, 3)  # a few generations
                    local_sigma = 0.1 * sigma  # smaller sigma
                    xmean_loc = xbest.copy()
                    pc_loc = np.zeros(d)
                    ps_loc = np.zeros(d)
                    C_loc = np.eye(d) + 1e-9 * np.eye(d)
                    for gen in range(local_gen_max):
                        if total_evals + local_lam > self.budget:
                            break
                        try:
                            B_loc = np.linalg.cholesky(C_loc)
                        except np.linalg.LinAlgError:
                            C_loc += 1e-9 * np.eye(d)
                            B_loc = np.linalg.cholesky(C_loc)
                        z_loc = self.rs.randn(local_lam, d)
                        x_raw_loc = xmean_loc + local_sigma * np.dot(z_loc, B_loc.T)
                        x_loc = x_raw_loc.copy()
                        for i in range(local_lam):
                            for j in range(d):
                                if x_loc[i, j] < lb[j]:
                                    x_loc[i, j] = lb[j] + (lb[j] - x_loc[i, j])
                                if x_loc[i, j] > ub[j]:
                                    x_loc[i, j] = ub[j] - (x_loc[i, j] - ub[j])
                                if x_loc[i, j] < lb[j] or x_loc[i, j] > ub[j]:
                                    x_loc[i, j] = np.clip(x_loc[i, j], lb[j], ub[j])
                        fvals_loc = np.empty(local_lam)
                        for i in range(local_lam):
                            fvals_loc[i] = func(x_loc[i])
                            total_evals += 1
                            if fvals_loc[i] < fbest:
                                fbest = fvals_loc[i]
                                xbest = x_loc[i].copy()
                                report_best(fbest, xbest)
                        # Update mean and covariance for local refinement
                        idx_loc = np.argsort(fvals_loc)
                        x_sorted_loc = x_loc[idx_loc]
                        old_xmean_loc = xmean_loc.copy()
                        xmean_loc = np.dot(weights[:local_lam//2], x_sorted_loc[:local_lam//2])  # using first half as parents
                        x_diff_loc = (xmean_loc - old_xmean_loc) / local_sigma
                        pc_loc = (1 - cc) * pc_loc + np.sqrt(cc * (2 - cc) * (local_lam//2)) * x_diff_loc
                        C_loc = (1 - c1 - cmu) * C_loc + c1 * np.outer(pc_loc, pc_loc)
                        diff_loc = x_sorted_loc[:local_lam//2] - old_xmean_loc
                        diff_norm_loc = diff_loc / local_sigma
                        C_loc += cmu * np.dot(diff_norm_loc.T, np.diag(weights[:local_lam//2]).dot(diff_norm_loc))
                        C_loc = (C_loc + C_loc.T) / 2.0
                        local_sigma *= np.exp((cs / ds) * (np.linalg.norm(ps_loc) / norm_expected - 1.0))
                    # end local generations
            # Restart: increase population
            popsize_multiplier *= 2
        return fbest, xbest