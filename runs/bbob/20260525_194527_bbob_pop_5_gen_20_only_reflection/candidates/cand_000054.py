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
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if eigvals[-1] / max(eigvals[0], 1e-15) > 1e7:
                        break
                except np.linalg.LinAlgError:
                    pass
                z = self.rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
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
                # Stagnation check with fitness diversity
                if no_improve_generations >= max_no_improve:
                    break
                if no_improve_generations > 10:
                    # Compute diversity (range of fvals)
                    f_range = np.max(fvals) - np.min(fvals)
                    if f_range < 1e-8 * (1 + np.abs(np.min(fvals))):
                        break

            # Local refinement using Nelder-Mead after CMA-ES run
            if total_evals < self.budget:
                # Use best point found in this CMA run
                x_start = xbest.copy()
                f_start = func(x_start)
                total_evals += 1
                # Nelder-Mead parameters
                simplex = [x_start.copy()]
                for i in range(d):
                    perturb = np.zeros(d)
                    perturb[i] = 0.05 * (ub[i] - lb[i])
                    new_point = np.clip(x_start + perturb, lb, ub)
                    simplex.append(new_point)
                fvals_nm = np.array([f_start] + [func(p) for p in simplex[1:]])
                total_evals += d
                
                # Ensure we haven't exceeded budget
                while total_evals < self.budget:
                    # Order by fval
                    order = np.argsort(fvals_nm)
                    simplex = [simplex[i] for i in order]
                    fvals_nm = fvals_nm[order]
                    # Compute centroid of best d points
                    centroid = np.mean(simplex[:d], axis=0)
                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    fr = func(xr)
                    total_evals += 1
                    if fvals_nm[0] <= fr < fvals_nm[-2]:
                        simplex[-1] = xr
                        fvals_nm[-1] = fr
                    elif fr < fvals_nm[0]:
                        # Expansion
                        xe = centroid + 2 * (xr - centroid)
                        xe = np.clip(xe, lb, ub)
                        fe = func(xe)
                        total_evals += 1
                        if fe < fr:
                            simplex[-1] = xe
                            fvals_nm[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals_nm[-1] = fr
                    else:
                        # Contraction
                        if fr < fvals_nm[-1]:
                            xc = centroid + 0.5 * (xr - centroid)
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            total_evals += 1
                            if fc < fr:
                                simplex[-1] = xc
                                fvals_nm[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, d+1):
                                    simplex[i] = 0.5 * (simplex[0] + simplex[i])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    fvals_nm[i] = func(simplex[i])
                                    total_evals += 1
                        else:
                            xc = centroid - 0.5 * (xr - centroid)
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            total_evals += 1
                            if fc < fvals_nm[-1]:
                                simplex[-1] = xc
                                fvals_nm[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, d+1):
                                    simplex[i] = 0.5 * (simplex[0] + simplex[i])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    fvals_nm[i] = func(simplex[i])
                                    total_evals += 1
                    # Update best
                    if fvals_nm[0] < fbest:
                        fbest = fvals_nm[0]
                        xbest = simplex[0].copy()
                        report_best(fbest, xbest)
                    # Check convergence or budget
                    if np.std(fvals_nm) < 1e-10 * (1 + np.abs(np.mean(fvals_nm))):
                        break
            popsize_multiplier *= 2
            if total_evals >= self.budget:
                break
        return fbest, xbest