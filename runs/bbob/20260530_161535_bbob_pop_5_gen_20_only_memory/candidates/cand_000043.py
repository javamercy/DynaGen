import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        domain_range = ub - lb
        dim = self.dim
        budget = self.budget
        rng = self.rng

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

        # Phase 1: Initial LHS (small budget)
        n_init = min(2 * dim, max(2, budget // 20))
        points = np.empty((n_init, dim))
        for i in range(dim):
            points[:, i] = rng.uniform(lb[i], ub[i], size=n_init)
        for i in range(dim):
            rng.shuffle(points[:, i])
        for i in range(n_init):
            if count >= budget:
                return best_f, best_x
            evaluate(points[i])
        if count >= budget:
            return best_f, best_x

        # Phase 2: CMA-ES (conservative exploitation)
        lam = max(4, int(3 * np.log(dim)))
        mu = lam // 2
        if mu < 1:
            mu = 1
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        sigma0 = 0.2 * np.mean(domain_range)
        xmean = best_x.copy()
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        invsqrtC = np.eye(dim)
        eigen_eval = 0

        cma_budget = int(budget * 0.4)
        cma_used = 0
        while count + lam <= budget and cma_used + lam <= cma_budget:
            arx = []
            arf = []
            for k in range(lam):
                if count >= budget or cma_used + k >= cma_budget:
                    break
                z = rng.normal(0, 1, dim)
                y = B @ (D * z)
                x = xmean + sigma0 * y
                x = np.clip(x, lb, ub)
                arx.append(x)
                f = evaluate(x)
                arf.append(f)
            if not arx:
                break
            count_used = len(arx)
            cma_used += count_used
            if count_used < lam:
                break

            idx = np.argsort(arf)
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * np.array(arx)[idx[:mu]], axis=0)

            dmean = xmean - xold
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ dmean / sigma0)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (dmean / sigma0)

            C *= (1 - c1 - cmu)
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                diff = (np.array(arx)[idx[i]] - xold) / sigma0
                C += cmu * weights[i] * np.outer(diff, diff)

            sigma0 *= np.exp((cs / damps) * (np.linalg.norm(ps) / (np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))) - 1))

            if count - eigen_eval > dim:
                eigen_eval = count
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eigh(C)
                D = np.abs(D)
                D = np.maximum(D, 1e-30)
                D = np.sqrt(D)
                invsqrtC = B @ np.diag(1/D) @ B.T

            if sigma0 < 1e-6 * np.mean(domain_range):
                break

        # Phase 3: Nelder-Mead with restarts (exploitation)
        if count < budget and best_x is not None:
            nm_budget = int((budget - count) * 0.7)
            nm_used = 0
            while count < budget and nm_used < nm_budget:
                # start simplex from best_x with random step
                step = 0.05 * domain_range + 1e-10
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = best_x.copy()
                for i in range(dim):
                    x = simplex[0].copy()
                    x[i] += step[i] * (1 + 0.1 * rng.randn())
                    x = np.clip(x, lb, ub)
                    simplex[i+1] = x

                f_vals = np.full(dim + 1, np.inf)
                for i in range(dim + 1):
                    if count >= budget:
                        break
                    f_vals[i] = evaluate(simplex[i])
                    nm_used += 1

                max_iter = min(100, (nm_budget - nm_used) // (dim + 1) + 1)
                for iteration in range(max_iter):
                    if count >= budget or nm_used >= nm_budget:
                        break
                    order = np.argsort(f_vals)
                    simplex = simplex[order]
                    f_vals = f_vals[order]

                    if f_vals[-1] - f_vals[0] < 1e-12:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + 1.0 * (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    fr = evaluate(xr)
                    nm_used += 1
                    if count >= budget or nm_used >= nm_budget:
                        break

                    if fr < f_vals[0]:
                        # Expansion
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        xe = np.clip(xe, lb, ub)
                        fe = evaluate(xe)
                        nm_used += 1
                        if fe < fr:
                            simplex[-1] = xe
                            f_vals[-1] = fe
                        else:
                            simplex[-1] = xr
                            f_vals[-1] = fr
                    elif fr < f_vals[-2]:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                    else:
                        # Contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, lb, ub)
                        fc = evaluate(xc)
                        nm_used += 1
                        if fc < f_vals[-1]:
                            simplex[-1] = xc
                            f_vals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                f_vals[i] = evaluate(simplex[i])
                                nm_used += 1
                                if count >= budget or nm_used >= nm_budget:
                                    break
                    idx_min = np.argmin(f_vals)
                    if f_vals[idx_min] < best_f:
                        best_f = f_vals[idx_min]
                        best_x = simplex[idx_min].copy()
                        report_best(best_f, best_x)
                # Restart after Nelder-Mead convergence
                if count < budget and nm_used < nm_budget:
                    # Perturb best_x to potentially escape local minima
                    pert = rng.uniform(-0.1, 0.1, size=dim) * domain_range
                    new_x = np.clip(best_x + pert, lb, ub)
                    evaluate(new_x)
                    nm_used += 1

        # Phase 4: Random local sampling
        if count < budget and best_x is not None:
            local_sigma = 0.01 * np.mean(domain_range)
            while count < budget:
                for _ in range(max(1, int((budget - count) / 5))):
                    if count >= budget:
                        break
                    x = best_x + rng.normal(0, local_sigma, size=dim)
                    evaluate(x)
                local_sigma *= 0.9
        elif count < budget:
            while count < budget:
                x = rng.uniform(lb, ub, size=dim)
                evaluate(x)

        return best_f, best_x