import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        domain_range = ub - lb
        dim = self.dim
        rng = np.random.RandomState(self.seed)
        total_count = 0
        best_f = np.inf
        best_x = None

        def evaluate(x):
            nonlocal total_count, best_f, best_x
            x = np.clip(x, lb, ub)
            f = func(x)
            total_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
            return f

        # Phase 1: CMA-ES
        budget_cma = max(1, int(0.8 * self.budget))
        xmean = rng.uniform(lb, ub, size=dim)
        xmean = np.clip(xmean, lb, ub)
        fmean = evaluate(xmean)
        if total_count >= self.budget:
            return best_f, best_x

        lam = 6 + int(3 * np.log(dim))
        lam = min(lam, budget_cma - total_count)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        cc = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim+1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((dim+2)**2 + mu_eff))
        damps = 1 + 2*max(0, np.sqrt((mu_eff-1)/(dim+1)) - 1) + cs
        sigma = 0.2 * np.mean(domain_range)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        invsqrtC = np.eye(dim)
        eigen_eval = 0
        local_count = 1

        while local_count + lam <= budget_cma and total_count < self.budget:
            arx = []
            arf = []
            for _ in range(lam):
                z = rng.normal(0, 1, dim)
                y = B @ (D * z)
                x = xmean + sigma * y
                x = np.clip(x, lb, ub)
                arx.append(x)
                f = evaluate(x)
                local_count += 1
                arf.append(f)
                if total_count >= self.budget:
                    break
            if total_count >= self.budget:
                break

            idx = np.argsort(arf)
            xold = xmean
            xmean = np.dot(weights, np.array(arx)[idx[:mu]])

            dmean = xmean - xold
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (invsqrtC @ dmean / sigma)
            pc = (1-cc)*pc + np.sqrt(cc*(2-cc)*mu_eff) * (dmean / sigma)

            C *= (1 - c1 - cmu)
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                diff = (np.array(arx)[idx[i]] - xold) / sigma
                C += cmu * weights[i] * np.outer(diff, diff)

            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/ (np.sqrt(dim)*(1-1/(4*dim)+1/(21*dim**2))) - 1))

            if local_count - eigen_eval > dim:
                eigen_eval = local_count
                C = np.triu(C) + np.triu(C,1).T
                D, B = np.linalg.eigh(C)
                D = np.abs(D)
                D = np.maximum(D, 1e-30)
                D = np.sqrt(D)
                invsqrtC = B @ np.diag(1/D) @ B.T

            if sigma < 1e-10 * np.mean(domain_range):
                break

        # Phase 2: Nelder-Mead
        if best_x is not None and total_count < self.budget:
            pert = 0.1 * np.mean(domain_range)
            simplex = [best_x.copy()]
            for i in range(dim):
                pt = best_x.copy()
                pt[i] += pert
                pt = np.clip(pt, lb, ub)
                simplex.append(pt)
            simplex_fvals = [best_f]
            for i in range(1, dim+1):
                if total_count >= self.budget:
                    break
                f = evaluate(simplex[i])
                simplex_fvals.append(f)
            order = np.argsort(simplex_fvals)
            simplex = [simplex[i] for i in order]
            simplex_fvals = [simplex_fvals[i] for i in order]

            while total_count < self.budget:
                if np.std(simplex_fvals) < 1e-12:
                    break
                xbar = np.mean(simplex[:-1], axis=0)
                xr = xbar + (xbar - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if total_count >= self.budget:
                    break
                fr = evaluate(xr)
                if fr < simplex_fvals[-2] and fr >= simplex_fvals[0]:
                    simplex[-1] = xr
                    simplex_fvals[-1] = fr
                elif fr < simplex_fvals[0]:
                    xe = xbar + 2*(xbar - simplex[-1])
                    xe = np.clip(xe, lb, ub)
                    if total_count >= self.budget:
                        break
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_fvals[-1] = fr
                else:
                    xc = xbar + 0.5*(simplex[-1] - xbar)
                    xc = np.clip(xc, lb, ub)
                    if total_count >= self.budget:
                        break
                    fc = evaluate(xc)
                    if fc < simplex_fvals[-1]:
                        simplex[-1] = xc
                        simplex_fvals[-1] = fc
                    else:
                        for i in range(1, dim+1):
                            simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            if total_count >= self.budget:
                                break
                            simplex_fvals[i] = evaluate(simplex[i])
                order = np.argsort(simplex_fvals)
                simplex = [simplex[i] for i in order]
                simplex_fvals = [simplex_fvals[i] for i in order]

        return best_f, best_x