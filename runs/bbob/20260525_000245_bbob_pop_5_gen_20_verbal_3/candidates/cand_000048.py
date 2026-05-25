import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        n = self.dim
        budget = self.budget

        # Initial feasible point
        mean = np.random.uniform(lb, ub, n)
        best_x = mean.copy()
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        if budget < 4:
            for _ in range(budget - calls):
                x = np.random.uniform(lb, ub, n)
                val = func(x)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x
                    report_best(best_val, best_x)
            return best_val, best_x

        # Allocate budget for CMA-ES (80%) and local refinement (20%)
        cma_budget = int(budget * 0.8)
        if cma_budget < 4:
            cma_budget = budget
        nm_budget = budget - cma_budget

        # CMA-ES parameters
        lambda_ = min(cma_budget - calls, 4 + int(4 * np.log(n)))
        lambda_ = max(2, lambda_)
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)

        c_s = (mu_eff + 2) / (n + mu_eff + 5)
        d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        if mu == 1:
            c_mu = 0.0

        sigma = 0.2 * np.mean(ub - lb)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        no_improve_iter = 0
        max_no_improve = max(5, int(cma_budget / (4 * lambda_)))

        while calls < cma_budget:
            if calls + lambda_ > cma_budget:
                lambda_actual = cma_budget - calls
            else:
                lambda_actual = lambda_
            if lambda_actual < 1:
                break

            try:
                samples = np.random.multivariate_normal(mean, sigma ** 2 * C, size=lambda_actual)
            except:
                samples = mean + sigma * np.random.randn(lambda_actual, n) * np.sqrt(np.diag(C))
            samples = np.clip(samples, lb, ub)

            vals = np.array([func(s) for s in samples])
            calls += lambda_actual

            idx = np.argsort(vals)
            vals_sorted = vals[idx]
            samples_sorted = samples[idx]

            if vals_sorted[0] < best_val:
                best_val = vals_sorted[0]
                best_x = samples_sorted[0]
                report_best(best_val, best_x)
                no_improve_iter = 0
            else:
                no_improve_iter += 1

            old_mean = mean.copy()
            mean = np.dot(w, samples_sorted[:mu])

            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T))
            except:
                invsqrtC = np.eye(n)

            ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * np.dot(invsqrtC, (mean - old_mean) / sigma)
            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma = sigma * np.exp((c_s / d_s) * (norm_ps / expected_norm - 1))

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

            diffs = (samples_sorted[:mu] - old_mean) / sigma
            C_mu = np.zeros((n, n))
            for i in range(mu):
                C_mu += w[i] * np.outer(diffs[i], diffs[i])
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * C_mu
            C = (C + C.T) / 2

            if np.linalg.det(C) <= 0:
                C = np.eye(n)

            restart = False
            if sigma < 1e-12 * np.mean(ub - lb):
                restart = True
            if no_improve_iter >= max_no_improve:
                restart = True

            if restart and calls < cma_budget:
                mean = best_x + 0.5 * np.random.uniform(-1, 1, n) * (ub - lb)
                mean = np.clip(mean, lb, ub)
                sigma = 0.3 * np.mean(ub - lb)
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                no_improve_iter = 0

        # Local refinement using Nelder-Mead on remaining budget
        if nm_budget > 0 and calls < budget:
            nm_evals = budget - calls
            # Build initial simplex around best_x
            simplex = np.zeros((n+1, n))
            simplex[0] = best_x
            for i in range(n):
                pert = 0.05 * (ub[i] - lb[i])
                if pert <= 0:
                    pert = 0.05
                simplex[i+1] = best_x.copy()
                simplex[i+1, i] += pert
                simplex[i+1] = np.clip(simplex[i+1], lb, ub)
            fvals = np.array([func(p) for p in simplex])
            calls += n+1
            # update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_val:
                best_val = fvals[idx_min]
                best_x = simplex[idx_min]
                report_best(best_val, best_x)

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            while calls < budget:
                # order
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                centroid = np.mean(simplex[:-1], axis=0)

                # reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                calls += 1
                if fr < best_val:
                    best_val = fr
                    best_x = xr
                    report_best(best_val, best_x)

                if fr < fvals[0]:
                    # expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    calls += 1
                    if fe < best_val:
                        best_val = fe
                        best_x = xe
                        report_best(best_val, best_x)
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
                    # contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    calls += 1
                    if fc < best_val:
                        best_val = fc
                        best_x = xc
                        report_best(best_val, best_x)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                        for i in range(1, n+1):
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if fvals[i] < best_val:
                                best_val = fvals[i]
                                best_x = simplex[i]
                                report_best(best_val, best_x)

        return best_val, best_x