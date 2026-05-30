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

        # initial random point
        x0 = rng.uniform(lb, ub, size=dim)
        evaluate(x0)

        # IPOP-CMA-ES
        lam_init = 4 + int(3 * np.log(dim))
        max_restarts = max(1, int(np.log(budget)))
        for restart in range(max_restarts):
            if count >= budget:
                break
            lam = lam_init * (2 ** restart)
            lam = min(lam, budget - count)
            if lam < 2:
                break
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2) ** 2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

            sigma = 0.5 * np.mean(domain_range)
            if best_x is not None and restart > 0:
                xmean = best_x.copy()
            else:
                xmean = rng.uniform(lb, ub, size=dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigen_eval = 0

            if count < budget:
                evaluate(xmean)

            while count + lam <= budget:
                arx = []
                arf = []
                for k in range(lam):
                    if count >= budget:
                        break
                    z = rng.normal(0, 1, dim)
                    y = B @ (D * z)
                    x = xmean + sigma * y
                    x = np.clip(x, lb, ub)
                    arx.append(x)
                    f = evaluate(x)
                    arf.append(f)
                if len(arx) < lam:
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

                if sigma < 1e-8 * np.mean(domain_range):
                    break

            # local Nelder-Mead search after CMA-ES
            if count < budget and best_x is not None:
                n_local = min(50, (budget - count) // (dim + 1))
                step = 0.02 * domain_range
                step[step == 0] = 1e-8
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = best_x
                for i in range(dim):
                    x = simplex[0].copy()
                    x[i] += step[i]
                    x = np.clip(x, lb, ub)
                    simplex[i+1] = x
                f_vals = np.array([evaluate(s) for s in simplex])
                for _ in range(n_local):
                    if count >= budget:
                        break
                    order = np.argsort(f_vals)
                    simplex = simplex[order]
                    f_vals = f_vals[order]
                    if f_vals[-1] - f_vals[0] < 1e-12:
                        break
                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + 1.0 * (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    fr = evaluate(xr)
                    if count >= budget:
                        break
                    if fr < f_vals[0]:
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        xe = np.clip(xe, lb, ub)
                        fe = evaluate(xe)
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
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, lb, ub)
                        fc = evaluate(xc)
                        if fc < f_vals[-1]:
                            simplex[-1] = xc
                            f_vals[-1] = fc
                        else:
                            for i in range(1, dim+1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                f_vals[i] = evaluate(simplex[i])
                            best_local = np.argmin(f_vals)
                            if f_vals[best_local] < best_f:
                                best_f = f_vals[best_local]
                                best_x = simplex[best_local].copy()
                                report_best(best_f, best_x)

        # remaining budget random sampling
        while count < budget:
            x = rng.uniform(lb, ub)
            evaluate(x)

        return best_f, best_x