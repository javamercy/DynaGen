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

        # Initial LHS
        n_init = min(2 * dim, max(2, budget // 10))
        points = np.empty((n_init, dim))
        for i in range(dim):
            points[:, i] = rng.uniform(lb[i], ub[i], size=n_init)
        for i in range(dim):
            rng.shuffle(points[:, i])
        for i in range(n_init):
            if count >= budget:
                break
            evaluate(points[i])

        # CMA-ES (exploration oriented, same as parent)
        lam = 8 + int(4 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff) * 0.75
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2) ** 2 + mu_eff)) * 0.75
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        max_restarts = max(1, int(budget / (lam * dim)))
        cma_budget = int((budget - count) * 0.8)
        if cma_budget < lam:
            cma_budget = budget - count
        cma_used = 0

        for restart in range(max_restarts + 1):
            if count >= budget:
                break
            if cma_used >= cma_budget:
                break
            sigma = 0.4 * np.mean(domain_range)
            xmean = best_x.copy() if best_x is not None else rng.uniform(lb, ub, size=dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigen_eval = 0

            if count < budget:
                evaluate(xmean)

            while count + lam <= budget and cma_used + lam <= cma_budget:
                arx = []
                arf = []
                for k in range(lam):
                    if count >= budget or cma_used + k >= cma_budget:
                        break
                    z = rng.normal(0, 1, dim)
                    y = B @ (D * z)
                    x = xmean + sigma * y
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

                if sigma < 1e-4 * np.mean(domain_range):
                    break

        # Pattern search local refinement (intensification)
        if best_x is not None:
            step = 0.1 * domain_range
            while count < budget:
                improved = False
                # Random order of coordinates
                coords = rng.permutation(dim)
                for i in coords:
                    if count >= budget:
                        break
                    # Positive direction
                    candidate = best_x.copy()
                    candidate[i] = np.clip(candidate[i] + step[i], lb[i], ub[i])
                    f_candidate = evaluate(candidate)
                    if f_candidate < best_f:
                        improved = True
                        continue  # best_x already updated
                    # Negative direction
                    candidate = best_x.copy()
                    candidate[i] = np.clip(candidate[i] - step[i], lb[i], ub[i])
                    f_candidate = evaluate(candidate)
                    if f_candidate < best_f:
                        improved = True
                # If no improvement in full cycle, reduce step
                if not improved:
                    step *= 0.5
                    # Also occasionally do a random jump to escape
                    if rng.rand() < 0.2:
                        candidate = best_x + rng.uniform(-0.2, 0.2, size=dim) * domain_range
                        candidate = np.clip(candidate, lb, ub)
                        evaluate(candidate)
                # Minimum step size
                step = np.maximum(step, 1e-6 * domain_range)
                # If step becomes too small, break
                if np.max(step) < 1e-8 * np.mean(domain_range):
                    break
        # Fallback if best_x is None (should not happen)
        if best_x is None:
            evaluate(rng.uniform(lb, ub))

        return best_f, best_x