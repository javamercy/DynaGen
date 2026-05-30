import numpy as np
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        range_ = ub - lb
        # Initial point
        x = np.random.uniform(lb, ub)
        best_val = func(x)
        best_x = x.copy()
        evals = 1
        report_best(best_val, best_x)
        if self.budget == 1:
            return best_val, best_x
        # Phase 1: Exploitative CMA-ES
        mean = best_x.copy()
        sigma = 0.15 * np.mean(range_)
        C = np.eye(dim)
        lam = max(2, 4 + int(2 * math.log(dim)))
        if lam > self.budget - evals:
            lam = max(2, self.budget - evals)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 * 4 / ((dim + 1.3)**2 + mueff)  # doubled
        cmu = 2 * 2 * min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))  # doubled
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        last_improve = evals
        stagnation_limit = max(10, int(0.05 * self.budget))
        switch_to_local = False
        while evals < self.budget and not switch_to_local:
            remaining = self.budget - evals
            if remaining < lam:
                lam = max(2, remaining)
                mu = lam // 2
                if mu > 0:
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= weights.sum()
                    mueff = 1.0 / np.sum(weights**2)
                    cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
                    cs = (mueff + 2) / (dim + mueff + 5)
                    c1 = 4 / ((dim + 1.3)**2 + mueff)
                    cmu = 4 * min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
                    damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                A = np.eye(dim)
            candidates = []
            for _ in range(lam):
                z = np.random.randn(dim)
                cand = mean + sigma * A @ z
                cand = np.clip(cand, lb, ub)
                candidates.append(cand)
            vals = []
            for cand in candidates:
                if evals >= self.budget:
                    break
                val = func(cand)
                vals.append(val)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = cand.copy()
                    report_best(best_val, best_x)
                    last_improve = evals
            if len(vals) == 0:
                break
            idx = np.argsort(vals)
            candidates = [candidates[i] for i in idx]
            old_mean = mean.copy()
            mean = np.sum([weights[i] * candidates[i] for i in range(mu)], axis=0)
            mean = np.clip(mean, lb, ub)
            z_mean = (mean - old_mean) / sigma
            try:
                invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                invsqrtC = np.eye(dim)
            ps = (1 - cs) * ps + math.sqrt(cs*(2-cs)*mueff) * invsqrtC @ z_mean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1-cs)**(2*evals/lam)) < (1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc*(2-cc)*mueff) * z_mean
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
            for i in range(mu):
                z = (candidates[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2
            sigma *= math.exp((cs/damps) * (np.linalg.norm(ps)/math.sqrt(dim) - 1))
            if sigma < 1e-10 * np.mean(range_):
                sigma = 0.01 * np.mean(range_)
                C = np.eye(dim)
                pc.fill(0)
                ps.fill(0)
            if evals - last_improve > stagnation_limit:
                switch_to_local = True
        # Phase 2: Nelder-Mead simplex local search
        if evals < self.budget:
            # Build simplex around best_x
            n = dim
            simplex = [best_x.copy()]
            for i in range(n):
                point = best_x.copy()
                step = 0.05 * range_[i] if range_[i] > 0 else 0.05
                point[i] = np.clip(point[i] + step, lb[i], ub[i])
                simplex.append(point)
            # Evaluate simplex vertices (skip first if already evaluated, but we re-evaluate for consistency)
            fvals = []
            for i, point in enumerate(simplex):
                if i == 0 and evals < self.budget:
                    # best_x already evaluated, but we need its value for comparison; we can reuse best_val
                    fvals.append(best_val)
                else:
                    if evals >= self.budget:
                        break
                    val = func(point)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = point.copy()
                        report_best(best_val, best_x)
                    fvals.append(val)
            if len(fvals) < n+1:
                fvals = [best_val] * (n+1)  # fallback, though unlikely
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            while evals < self.budget:
                # Order simplex by fvals (ascending)
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                # Centroid of best n points
                centroid = np.mean(simplex[:n], axis=0)
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if evals >= self.budget:
                    break
                fr = func(xr)
                evals += 1
                if fr < best_val:
                    best_val = fr
                    best_x = xr.copy()
                    report_best(best_val, best_x)
                if fvals[0] <= fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if evals >= self.budget:
                        break
                    fe = func(xe)
                    evals += 1
                    if fe < best_val:
                        best_val = fe
                        best_x = xe.copy()
                        report_best(best_val, best_x)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, lb, ub)
                    if evals >= self.budget:
                        break
                    fc = func(xc)
                    evals += 1
                    if fc < best_val:
                        best_val = fc
                        best_x = xc.copy()
                        report_best(best_val, best_x)
                    if fc < min(fvals[-1], fr):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            if evals >= self.budget:
                                break
                            fvals[i] = func(simplex[i])
                            evals += 1
                            if fvals[i] < best_val:
                                best_val = fvals[i]
                                best_x = simplex[i].copy()
                                report_best(best_val, best_x)
                # Check for convergence (optional, but not necessary)
                if np.std(fvals) < 1e-12:
                    break
        return best_val, best_x