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

        # CMA-ES parameters
        lam = 4 + int(4 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2) ** 2 + mu_eff))
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

        # Initial random point
        x0 = rng.uniform(lb, ub, size=dim)
        evaluate(x0)

        # Restart loop (exploration phase)
        max_restarts = max(1, int(budget / (10 * dim)))
        for restart in range(max_restarts + 1):
            if count >= budget:
                break
            sigma = 0.3 * np.mean(domain_range)
            xmean = rng.uniform(lb, ub, size=dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigen_eval = 0

            f = evaluate(xmean)
            if count >= budget:
                break

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

                if sigma < 1e-6 * np.mean(domain_range):
                    break

        # Local refinement with bounded Nelder-Mead
        if count < budget and best_x is not None:
            # Initialize simplex around best_x
            simplex = [best_x.copy()]
            for i in range(dim):
                p = best_x.copy()
                p[i] += 0.05 * domain_range[i]
                p = np.clip(p, lb, ub)
                simplex.append(p)
                if count >= budget:
                    break
            # Evaluate initial simplex (if not already evaluated)
            simplex_vals = []
            for i, p in enumerate(simplex):
                if i == 0:
                    # Use best_f directly
                    simplex_vals.append(best_f)
                else:
                    if count >= budget:
                        break
                    f = evaluate(p)
                    simplex_vals.append(f)
            if count < budget:
                # Nelder-Mead main loop
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_s = 0.5
                while count < budget:
                    # Sort simplex by function value
                    order = np.argsort(simplex_vals)
                    simplex = [simplex[i] for i in order]
                    simplex_vals = [simplex_vals[i] for i in order]
                    # Compute centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    if count >= budget:
                        break
                    fr = evaluate(xr)
                    if fr < simplex_vals[0]:
                        # Expansion
                        xe = centroid + gamma * (xr - centroid)
                        xe = np.clip(xe, lb, ub)
                        if count >= budget:
                            # Replace worst with xr
                            simplex[-1] = xr
                            simplex_vals[-1] = fr
                            break
                        fe = evaluate(xe)
                        if fe < simplex_vals[0]:
                            simplex[-1] = xe
                            simplex_vals[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_vals[-1] = fr
                    else:
                        if fr < simplex_vals[-2]:
                            simplex[-1] = xr
                            simplex_vals[-1] = fr
                        else:
                            # Contraction
                            if fr < simplex_vals[-1]:
                                xc = centroid + rho * (xr - centroid)
                            else:
                                xc = centroid + rho * (simplex[-1] - centroid)
                            xc = np.clip(xc, lb, ub)
                            if count >= budget:
                                break
                            fc = evaluate(xc)
                            if fc < simplex_vals[-1]:
                                simplex[-1] = xc
                                simplex_vals[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, len(simplex)):
                                    simplex[i] = simplex[0] + sigma_s * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    if count >= budget:
                                        break
                                    simplex_vals[i] = evaluate(simplex[i])
                    if count >= budget:
                        break
        return best_f, best_x