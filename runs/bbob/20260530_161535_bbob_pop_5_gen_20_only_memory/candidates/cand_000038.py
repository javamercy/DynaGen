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

        max_restarts = max(1, int(budget / (5 * dim)))
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
            stagnation_counter = 0

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

                # Check for stagnation: no improvement in best_f for this generation
                # Actually we compare best_f before and after generation
                # We'll track if best_f improved in the last lam evaluations
                # Simple: if the best_f after this generation is the same as before, increment counter
                if best_f == best_f_prev:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                best_f_prev = best_f

                # If stagnation for 2 generations, diversify
                if stagnation_counter >= 2:
                    # Replace mean with a random point
                    xmean = rng.uniform(lb, ub, size=dim)
                    sigma = 0.3 * np.mean(domain_range)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    invsqrtC = np.eye(dim)
                    stagnation_counter = 0
                    # Evaluate the new mean
                    f = evaluate(xmean)
                    if count >= budget:
                        break

            # End of CMA-ES loop for this restart

        # Remaining budget: uniform random sampling
        while count < budget:
            x = rng.uniform(lb, ub)
            evaluate(x)

        return best_f, best_x