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
            # initialize CMA-ES state with mixture
            if self.rs.rand() < 0.5:
                xmean = xbest.copy() + 0.1 * self.rs.randn(d) * (ub - lb)
            else:
                xmean = self.rs.uniform(lb, ub, d)
            xmean = np.clip(xmean, lb, ub)
            sigma = 0.1 * np.mean(ub - lb)
            C = np.eye(d)
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
            max_no_improve = 50 + int(np.floor(0.2 * d))
            condition_threshold = 1e7

            while total_evals + lam <= self.budget:
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                z = self.rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
                # resampling with bound constraints
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
                # stagnation check
                if np.min(fvals) < best_in_run:
                    best_in_run = np.min(fvals)
                    no_improve_generations = 0
                else:
                    no_improve_generations += 1
                # selection and update with Cauchy noise
                idx = np.argsort(fvals)
                x_sorted = x[idx]
                old_xmean = xmean.copy()
                xmean = np.dot(weights, x_sorted[:mu])
                # Cauchy noise to mean
                cauchy = self.rs.standard_cauchy(d)
                cauchy = np.clip(cauchy, -5, 5)  # avoid extreme jumps
                xmean += 0.1 * sigma * cauchy
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
                # condition number check
                if total_evals > 10 and no_improve_generations > 0:
                    eigvals = np.linalg.eigh(C)[0]
                    cond = eigvals[-1] / max(eigvals[0], 1e-12)
                    if cond > condition_threshold:
                        break
                if no_improve_generations >= max_no_improve:
                    break
            # local refinement after restart on best
            if total_evals < self.budget:
                # Nelder-Mead for up to 20 evaluations
                nm_budget = min(20, self.budget - total_evals)
                if nm_budget >= 4:
                    # build initial simplex around xbest
                    vertices = [xbest.copy()]
                    for i in range(d):
                        perturb = np.zeros(d)
                        perturb[i] = 0.05 * (ub[i] - lb[i])
                        v = xbest + perturb
                        v = np.clip(v, lb, ub)
                        vertices.append(v)
                    # evaluate vertices
                    fvals_simplex = []
                    for v in vertices:
                        fv = func(v)
                        total_evals += 1
                        if fv < fbest:
                            fbest = fv
                            xbest = v.copy()
                            report_best(fbest, xbest)
                        fvals_simplex.append(fv)
                    nm_used = len(vertices)
                    # Nelder-Mead iterations
                    while nm_used < nm_budget:
                        # order vertices
                        order = np.argsort(fvals_simplex)
                        vertices = [vertices[i] for i in order]
                        fvals_simplex = [fvals_simplex[i] for i in order]
                        # centroid of best d vertices
                        c = np.mean(vertices[:-1], axis=0)
                        # reflect worst
                        xr = c + (c - vertices[-1])
                        xr = np.clip(xr, lb, ub)
                        fr = func(xr)
                        total_evals += 1
                        nm_used += 1
                        if fr < fbest:
                            fbest = fr
                            xbest = xr.copy()
                            report_best(fbest, xbest)
                        if fr < fvals_simplex[0]:
                            # expand
                            xe = c + 2.0 * (xr - c)
                            xe = np.clip(xe, lb, ub)
                            fe = func(xe)
                            total_evals += 1
                            nm_used += 1
                            if fe < fr:
                                vertices[-1] = xe
                                fvals_simplex[-1] = fe
                            else:
                                vertices[-1] = xr
                                fvals_simplex[-1] = fr
                        elif fr < fvals_simplex[-2]:
                            vertices[-1] = xr
                            fvals_simplex[-1] = fr
                        else:
                            # contract
                            xc = c + 0.5 * (vertices[-1] - c)
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            total_evals += 1
                            nm_used += 1
                            if fc < fvals_simplex[-1]:
                                vertices[-1] = xc
                                fvals_simplex[-1] = fc
                            else:
                                # shrink
                                for i in range(1, len(vertices)):
                                    xi = vertices[0] + 0.5 * (vertices[i] - vertices[0])
                                    xi = np.clip(xi, lb, ub)
                                    fi = func(xi)
                                    total_evals += 1
                                    nm_used += 1
                                    if fi < fbest:
                                        fbest = fi
                                        xbest = xi.copy()
                                        report_best(fbest, xbest)
                                    vertices[i] = xi
                                    fvals_simplex[i] = fi
                        if total_evals >= self.budget:
                            break
            popsize_multiplier *= 2
            if total_evals >= self.budget:
                break
        return fbest, xbest