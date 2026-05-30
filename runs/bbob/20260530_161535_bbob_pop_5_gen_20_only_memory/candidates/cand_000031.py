import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)

        lam = max(2, 2 + int(np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2.4 / ((dim + 1.3) ** 2 + mu_eff) * 1.5
        cmu = min(1 - c1, 2.4 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff)) * 1.5
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        best_x = None
        best_f = np.inf
        count = 0

        def evaluate(x):
            nonlocal count, best_x, best_f
            x = np.clip(x, lb, ub)
            f = func(x)
            count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(f, best_x)
            return f

        sigma = 0.1 * np.mean(ub - lb)
        xmean = rng.uniform(lb, ub, size=dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        invsqrtC = np.eye(dim)
        eigen_eval = 0

        evaluate(xmean)
        if count >= budget:
            return best_f, best_x

        while count + lam <= budget:
            arx = []
            arf = []
            for k in range(lam):
                z = rng.normal(0, 1, dim)
                y = B @ (D * z)
                x = xmean + sigma * y
                x = np.clip(x, lb, ub)
                arx.append(x)
                f = evaluate(x)
                arf.append(f)
                if count >= budget:
                    break
            if count >= budget:
                break

            idx = np.argsort(arf)
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * np.array(arx)[idx[:mu]], axis=0)

            dmean = xmean - xold
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ dmean / sigma)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (dmean / sigma)

            C *= (1 - c1 - cmu)
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                diff = (np.array(arx)[idx[i]] - xold) / sigma
                C += cmu * weights[i] * np.outer(diff, diff)

            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / (np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))) - 1))

            if count - eigen_eval > dim:
                eigen_eval = count
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eigh(C)
                D = np.abs(D)
                D = np.maximum(D, 1e-30)
                D = np.sqrt(D)
                invsqrtC = B @ np.diag(1/D) @ B.T

            if sigma < 1e-10 * np.mean(ub - lb):
                break

        # Nelder-Mead local search
        remaining = budget - count
        if remaining > 0 and best_x is not None:
            # Initialize simplex
            simplex = [best_x.copy()]
            s = np.mean(ub - lb) * 0.01
            for i in range(dim):
                point = best_x.copy()
                point[i] = np.clip(point[i] + s, lb[i], ub[i])
                simplex.append(point)
            fvals = []
            for p in simplex:
                if count >= budget:
                    break
                f = evaluate(p)
                fvals.append(f)
            # Sort by fval
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            while count < budget:
                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if count >= budget:
                    break
                fr = evaluate(xr)
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if count >= budget:
                        break
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    if count >= budget:
                        break
                    fc = evaluate(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            if count >= budget:
                                break
                            fvals[i] = evaluate(simplex[i])
                # Re-sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                # Check convergence (optional)
                if np.max(simplex[0] - simplex[-1]) < 1e-12:
                    break

        return best_f, best_x