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
        # initial point
        x = self.rs.uniform(lb, ub, d)
        fbest = func(x)
        xbest = x.copy()
        report_best(fbest, xbest)
        total_evals = 1

        popsize_multiplier = 1
        while total_evals < self.budget:
            lam = (4 + int(3 * np.log(d))) * popsize_multiplier
            if total_evals + lam > self.budget:
                break
            # CMA-ES state initialization
            xmean = self.rs.uniform(lb, ub, d)
            sigma = 0.3 * np.mean(ub - lb)
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
            max_no_improve = 50 + int(0.2 * d)

            while total_evals + lam <= self.budget:
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                # Check eigenvalue ratio
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if eigvals[-1] / max(eigvals[0], 1e-15) > 1e7:
                        break
                except np.linalg.LinAlgError:
                    pass
                z = self.rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
                # resampling for bound constraints
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
                # evaluate
                fvals = np.empty(lam)
                for i in range(lam):
                    fvals[i] = func(x[i])
                    total_evals += 1
                    if fvals[i] < fbest:
                        fbest = fvals[i]
                        xbest = x[i].copy()
                        report_best(fbest, xbest)
                # stagnation and diversity check
                if np.min(fvals) < best_in_run:
                    best_in_run = np.min(fvals)
                    no_improve_generations = 0
                else:
                    no_improve_generations += 1
                # selection and update
                idx = np.argsort(fvals)
                x_sorted = x[idx]
                fvals_sorted = fvals[idx]
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
                # diversity stagnation: if top mu range is very small, break
                if no_improve_generations > 0 and mu > 1:
                    range_top_mu = fvals_sorted[mu-1] - fvals_sorted[0]
                    if range_top_mu < 1e-8 * (abs(fvals_sorted[0]) + 1e-10):
                        break
                if no_improve_generations >= max_no_improve:
                    break
            # local Nelder-Mead refinement after each restart
            if total_evals < self.budget:
                budget_left = self.budget - total_evals
                # use at most min(100, budget_left) evaluations for NM
                max_nm_evals = min(100, budget_left)
                if max_nm_evals >= d + 2:  # need at least simplex size
                    # initialize simplex from best point so far
                    x0 = xbest.copy()
                    sigma0 = 0.1 * np.mean(ub - lb)
                    n = d
                    simplex = np.array([x0 + sigma0 * self.rs.randn(n) for _ in range(n+1)])
                    for i in range(n+1):
                        simplex[i] = np.clip(simplex[i], lb, ub)
                    fvals_sim = np.array([func(simplex[i]) for i in range(n+1)])
                    total_evals += n+1
                    # update best
                    for i in range(n+1):
                        if fvals_sim[i] < fbest:
                            fbest = fvals_sim[i]
                            xbest = simplex[i].copy()
                            report_best(fbest, xbest)
                    # sort
                    idx = np.argsort(fvals_sim)
                    simplex = simplex[idx]
                    fvals_sim = fvals_sim[idx]
                    nm_evals = n+1
                    while nm_evals < max_nm_evals:
                        x0 = np.mean(simplex[:-1], axis=0)
                        # reflection
                        xr = x0 + (x0 - simplex[-1])
                        xr = np.clip(xr, lb, ub)
                        fr = func(xr)
                        nm_evals += 1
                        if fr < fbest:
                            fbest = fr
                            xbest = xr.copy()
                            report_best(fbest, xbest)
                        if fr < fvals_sim[0]:
                            # expansion
                            xe = x0 + 2*(xr - x0)
                            xe = np.clip(xe, lb, ub)
                            fe = func(xe)
                            nm_evals += 1
                            if fe < fbest:
                                fbest = fe
                                xbest = xe.copy()
                                report_best(fbest, xbest)
                            if fe < fr:
                                simplex[-1] = xe
                                fvals_sim[-1] = fe
                            else:
                                simplex[-1] = xr
                                fvals_sim[-1] = fr
                        elif fr < fvals_sim[-2]:
                            simplex[-1] = xr
                            fvals_sim[-1] = fr
                        else:
                            # contraction
                            if fr < fvals_sim[-1]:
                                xc = x0 + 0.5*(xr - x0)
                            else:
                                xc = x0 + 0.5*(simplex[-1] - x0)
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            nm_evals += 1
                            if fc < fbest:
                                fbest = fc
                                xbest = xc.copy()
                                report_best(fbest, xbest)
                            if fc < min(fvals_sim[-1], fr):
                                simplex[-1] = xc
                                fvals_sim[-1] = fc
                            else:
                                # shrink
                                for i in range(1, n+1):
                                    simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    fvals_sim[i] = func(simplex[i])
                                    nm_evals += 1
                                    if fvals_sim[i] < fbest:
                                        fbest = fvals_sim[i]
                                        xbest = simplex[i].copy()
                                        report_best(fbest, xbest)
                        # re-sort
                        idx = np.argsort(fvals_sim)
                        simplex = simplex[idx]
                        fvals_sim = fvals_sim[idx]
                        # early break if simplex very small
                        if np.std(simplex, axis=0).max() < 1e-8 * np.mean(ub - lb):
                            break
                    total_evals += nm_evals - (n+1)  # already added initial evaluations
            popsize_multiplier *= 2
            if total_evals >= self.budget:
                break
        return fbest, xbest