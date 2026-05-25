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
        evals = 1
        popsize_multiplier = 1
        initial_sigma = 0.5 * np.mean(ub - lb)

        while evals < self.budget:
            lam = (4 + int(3 * np.log(d))) * popsize_multiplier
            if evals + lam > self.budget:
                break
            # Initialize CMA-ES state
            if fbest < np.inf:
                xmean = xbest + self.rs.uniform(-0.1, 0.1, d) * (ub - lb)
                xmean = np.clip(xmean, lb, ub)
            else:
                xmean = self.rs.uniform(lb, ub, d)
            sigma = initial_sigma
            C = np.eye(d)
            mu = max(lam // 2, 1)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = 4.0 / (d + 4.0)
            cs = (mueff + 2.0) / (d + mueff + 5.0)
            c1 = 2.0 / ((d + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0/mueff) / ((d + 2.0)**2 + mueff))
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0)/(d + 1.0)) - 1.0) + cs
            pc = np.zeros(d)
            ps = np.zeros(d)
            norm_expected = np.sqrt(d) * (1.0 - 1.0/(4.0*d) + 1.0/(21.0*d*d))
            best_in_run = fbest
            no_improve = 0
            max_no_improve = 50 + int(0.2 * d)
            cond_threshold = 1e14

            while evals + lam <= self.budget:
                # Condition number check
                try:
                    C_eig = np.linalg.eigh(C)[0]
                    cond = C_eig[-1] / max(C_eig[0], 1e-20)
                except:
                    cond = 1.0
                if cond > cond_threshold:
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
                # Resample and mirror for bounds
                x = np.empty_like(x_raw)
                for i in range(lam):
                    xi = x_raw[i]
                    attempts = 0
                    while np.any(xi < lb) or np.any(xi > ub):
                        attempts += 1
                        if attempts > 10:
                            # Mirror
                            xi = np.where(xi < lb, 2*lb - xi, xi)
                            xi = np.where(xi > ub, 2*ub - xi, xi)
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
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * x_diff
                invB = np.linalg.solve(B, np.eye(d))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invB, x_diff)
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                diff = x_sorted[:mu] - old_xmean
                diff_norm = diff / sigma
                C += cmu * np.dot(diff_norm.T, np.dot(np.diag(weights), diff_norm))
                C = (C + C.T) / 2.0
                ps_norm = np.linalg.norm(ps)
                sigma *= np.exp((cs/ds) * (ps_norm / norm_expected - 1.0))
            # End of CMA-ES generation loop

            # Local refinement after restart
            if evals < self.budget:
                lam_local = max(4, int(3 * np.log(d)))
                sigma_local = 0.01 * initial_sigma
                if evals + lam_local <= self.budget:
                    # Initialize local CMA-ES from best point
                    xmean_loc = xbest + self.rs.uniform(-0.01, 0.01, d) * (ub - lb)
                    xmean_loc = np.clip(xmean_loc, lb, ub)
                    C_loc = np.eye(d)
                    pc_loc = np.zeros(d)
                    ps_loc = np.zeros(d)
                    for _ in range(2):  # two generations
                        if evals + lam_local > self.budget:
                            break
                        try:
                            B_loc = np.linalg.cholesky(C_loc)
                        except np.linalg.LinAlgError:
                            C_loc += 1e-9 * np.eye(d)
                            B_loc = np.linalg.cholesky(C_loc)
                        z_loc = self.rs.randn(lam_local, d)
                        x_raw_loc = xmean_loc + sigma_local * np.dot(z_loc, B_loc.T)
                        x_loc = np.empty_like(x_raw_loc)
                        for i in range(lam_local):
                            xi = x_raw_loc[i]
                            attempts = 0
                            while np.any(xi < lb) or np.any(xi > ub):
                                attempts += 1
                                if attempts > 10:
                                    xi = np.where(xi < lb, 2*lb - xi, xi)
                                    xi = np.where(xi > ub, 2*ub - xi, xi)
                                    xi = np.clip(xi, lb, ub)
                                    break
                                zi = self.rs.randn(d)
                                xi = xmean_loc + sigma_local * np.dot(zi, B_loc.T)
                            x_loc[i] = xi
                        fvals_loc = np.empty(lam_local)
                        for i in range(lam_local):
                            fvals_loc[i] = func(x_loc[i])
                            evals += 1
                            if fvals_loc[i] < fbest:
                                fbest = fvals_loc[i]
                                xbest = x_loc[i].copy()
                                report_best(fbest, xbest)
                        # Selection and update (simple, using same weights as main)
                        idx_loc = np.argsort(fvals_loc)
                        x_sorted_loc = x_loc[idx_loc]
                        xmean_loc = np.dot(weights[:mu], x_sorted_loc[:mu])
                        diff_loc = x_sorted_loc[:mu] - xmean_loc
                        C_loc = (1 - c1 - cmu) * C_loc + c1 * np.outer(pc_loc, pc_loc)
                        C_loc += cmu * np.dot(diff_loc.T, np.dot(np.diag(weights[:mu]), diff_loc)) / (sigma_local**2)
                        C_loc = (C_loc + C_loc.T) / 2.0
                        # No step-size adaptation for local refinement
            popsize_multiplier *= 2
        return fbest, xbest