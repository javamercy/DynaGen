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

        # CMA-ES parameters (more exploitative: smaller population)
        lam = max(4, int(4 + 2 * np.log(dim)))
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
            return f

        # Allocate budget: 70% for CMA-ES, 30% for local refinement
        cma_budget = int(0.7 * budget)
        local_budget = budget - cma_budget

        # CMA-ES initialization
        sigma = 0.3 * np.mean(domain_range)
        xmean = rng.uniform(lb, ub, size=dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        invsqrtC = np.eye(dim)
        eigen_eval = 0

        # Evaluate initial mean
        f = evaluate(xmean)
        if count >= cma_budget:
            pass
        else:
            # Main CMA-ES loop
            while count + lam <= cma_budget:
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
                    if count >= cma_budget:
                        break
                if count >= cma_budget:
                    break

                # Selection and recombination
                idx = np.argsort(arf)
                xold = xmean.copy()
                xmean = np.sum(weights[:, None] * np.array(arx)[idx[:mu]], axis=0)

                # Update paths
                dmean = xmean - xold
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ dmean / sigma)
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (dmean / sigma)

                # Update covariance
                C *= (1 - c1 - cmu)
                C += c1 * np.outer(pc, pc)
                for i in range(mu):
                    diff = (np.array(arx)[idx[i]] - xold) / sigma
                    C += cmu * weights[i] * np.outer(diff, diff)

                # Step-size adaptation
                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / (np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))) - 1))

                # Eigen decomposition
                if count - eigen_eval > dim:
                    eigen_eval = count
                    C = np.triu(C) + np.triu(C, 1).T
                    D, B = np.linalg.eigh(C)
                    D = np.abs(D)
                    D = np.maximum(D, 1e-30)
                    D = np.sqrt(D)
                    invsqrtC = B @ np.diag(1/D) @ B.T

            # After CMA-ES, if we used less than cma_budget, continue with remaining until cma_budget
            while count < cma_budget:
                x = rng.uniform(lb, ub)
                evaluate(x)

        # Local refinement phase
        if best_x is not None:
            sigma_local = 0.1 * np.mean(domain_range)
            decay = 0.95
            for _ in range(local_budget):
                if count >= budget:
                    break
                # Sample around best with decreasing step
                x = best_x + sigma_local * rng.randn(dim)
                x = np.clip(x, lb, ub)
                f = evaluate(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    sigma_local *= 1.1  # increase step if improvement
                else:
                    sigma_local *= decay
                sigma_local = max(sigma_local, 1e-10 * np.mean(domain_range))

        return best_f, best_x