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
        dim = self.dim
        budget = self.budget

        # Initial point
        best_x = np.random.uniform(lb, ub, dim)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        # CMA-ES parameters
        lambda_ = max(2, min(budget // 10, 4 + int(3 * np.log(dim))))
        if lambda_ > budget:
            lambda_ = budget
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize state
        mean = best_x.copy()
        sigma = 0.3 * np.mean(ub - lb)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        # Eigendecomposition for sampling and inverse sqrt
        evals, evecs = np.linalg.eigh(C)
        D = np.sqrt(evals)  # eigenvalues of sqrt(C)
        B = evecs

        calls_since_improvement = 0
        restart_threshold = budget // 5

        while calls < budget:
            # Determine actual population size for this generation
            lambda_actual = min(lambda_, budget - calls)
            if lambda_actual < 1:
                break

            # Sample points
            arz = np.random.randn(lambda_actual, dim)
            arx = mean + sigma * (arz @ (B * D).T)
            arx = np.clip(arx, lb, ub)

            # Evaluate
            arf = np.zeros(lambda_actual)
            for i in range(lambda_actual):
                arf[i] = func(arx[i])
            calls += lambda_actual

            # Sort
            idx = np.argsort(arf)
            arx_sort = arx[idx]
            arf_sort = arf[idx]

            # Update best
            if arf_sort[0] < best_val:
                best_val = arf_sort[0]
                best_x = arx_sort[0].copy()
                report_best(best_val, best_x)
                calls_since_improvement = 0
            else:
                calls_since_improvement += lambda_actual

            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * arx_sort[:mu], axis=0)

            # Update pc and ps
            step = (mean - old_mean) / sigma
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * step

            # Compute invsqrtC from eigendecomposition
            invsqrtC = B @ np.diag(1.0 / D) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ step)

            # Update C (rank-mu + rank-one)
            y = (arx_sort[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * (y[:, :, np.newaxis] * y[:, np.newaxis, :]), axis=0)

            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

            # Update eigendecomposition of C
            evals, evecs = np.linalg.eigh(C)
            D = np.sqrt(evals)
            B = evecs

            # Restart if no improvement for too long and budget left
            if calls_since_improvement >= restart_threshold and calls < budget - lambda_:
                mean = np.random.uniform(lb, ub, dim)
                sigma = 0.3 * np.mean(ub - lb)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                C = np.eye(dim)
                evals, evecs = np.linalg.eigh(C)
                D = np.sqrt(evals)
                B = evecs
                calls_since_improvement = 0

        return best_val, best_x