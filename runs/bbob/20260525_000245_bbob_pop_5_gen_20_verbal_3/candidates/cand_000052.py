import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        rng = self.rng

        # Population size
        lam = max(4, min(int(4 + 3 * np.log(dim)), budget // (dim + 1)))
        if lam < 4:
            lam = 4

        # Initialize mean and step-size
        xmean = rng.uniform(lb, ub, size=dim)
        sigma = 0.5 * (ub - lb).mean()
        best_x = xmean.copy()
        best_val = np.inf
        calls = 0

        # Evaluate initial mean
        if calls < budget:
            x = np.clip(xmean, lb, ub)
            val = func(x)
            calls += 1
            best_val = val
            best_x = x.copy()
            report_best(best_val, best_x)

        # Evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        invsqrtC = np.eye(dim)
        eigeneval = 0

        # Strategy constants
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # Restart parameters
        max_gen_no_improve = 20 + int(30 * dim / np.sqrt(lam))
        gen_no_improve = 0
        gen = 0
        tolerance = 1e-12

        while calls < budget:
            # Generate offspring
            arx = np.zeros((lam, dim))
            arfitness = np.full(lam, np.inf)
            for k in range(lam):
                if calls >= budget:
                    break
                arx[k] = xmean + sigma * B @ (D * rng.normal(size=dim))
                arx[k] = np.clip(arx[k], lb, ub)
                arfitness[k] = func(arx[k])
                calls += 1
                if arfitness[k] < best_val:
                    best_val = arfitness[k]
                    best_x = arx[k].copy()
                    report_best(best_val, best_x)
                    gen_no_improve = 0
                else:
                    gen_no_improve += 1

            # Sort by fitness
            idx = np.argsort(arfitness)
            arx = arx[idx]
            arfitness = arfitness[idx]

            # Update mean
            xold = xmean.copy()
            xmean = np.dot(weights, arx[:mu])

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * calls / lam)) < 1.4 + 2 / (dim + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma

            # Update covariance matrix
            artmp = (arx[:mu] - xold) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * np.dot(weights * artmp.T, artmp)

            # Update step-size
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

            # Eigen decomposition
            if calls - eigeneval > lam / (c1 + cmu) / dim / 10:
                eigeneval = calls
                try:
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-20))
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except np.linalg.LinAlgError:
                    pass

            # Check for restart: fitness diversity or no improvement
            if gen > 1:
                fitness_std = np.std(arfitness)
                fitness_range = np.max(arfitness) - np.min(arfitness)
                if fitness_range == 0:
                    diversity = 0
                else:
                    diversity = fitness_std / fitness_range
                if diversity < 1e-12 or gen_no_improve > max_gen_no_improve:
                    # Local refinement: sample around best
                    for _ in range(min(4 * dim, budget - calls)):
                        if calls >= budget:
                            break
                        perturbation = 0.1 * sigma * rng.normal(size=dim)
                        candidate = np.clip(best_x + perturbation, lb, ub)
                        val = func(candidate)
                        calls += 1
                        if val < best_val:
                            best_val = val
                            best_x = candidate.copy()
                            report_best(best_val, best_x)
                    # Restart: reinitialize mean and step-size
                    xmean = rng.uniform(lb, ub, size=dim)
                    sigma = 0.5 * (ub - lb).mean()
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    invsqrtC = np.eye(dim)
                    gen_no_improve = 0
                    # Evaluate new mean
                    if calls < budget:
                        x = np.clip(xmean, lb, ub)
                        val = func(x)
                        calls += 1
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
            gen += 1

        return best_val, best_x