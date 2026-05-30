import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget

        # Initial evaluation
        best_val = np.inf
        best_x = np.zeros(dim)
        x0 = np.random.uniform(lb, ub, size=dim)
        val0 = func(x0)
        best_val = val0
        best_x = x0
        report_best(best_val, best_x)
        evals = 1

        # Initial population size
        lam0 = max(2, int(4 + 3 * np.log(dim)))
        sigma0 = 0.2 * np.mean(ub - lb)
        lam = lam0
        restart = 0

        while evals < budget:
            if restart == 0:
                mean = best_x.copy()
                sigma = sigma0
                lam = lam0
            else:
                lam = int(lam * 2)
                mean = best_x.copy()
                sigma = sigma0

            # Adjust lambda to remaining budget
            if lam > budget - evals:
                lam = budget - evals
            if lam < 2:
                break

            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            generation = 0
            no_improve = 0

            while evals + lam <= budget:
                generation += 1
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.diag(np.sqrt(eigvals))
                B = eigvecs
                Z = np.random.randn(dim, lam)
                X = mean[:, np.newaxis] + sigma * (B @ D @ Z)
                X = np.clip(X, lb[:, np.newaxis], ub[:, np.newaxis])

                F = np.empty(lam)
                for i in range(lam):
                    F[i] = func(X[:, i])
                    evals += 1

                idx = np.argsort(F)
                F = F[idx]
                X = X[:, idx]

                if F[0] < best_val:
                    best_val = F[0]
                    best_x = X[:, 0].copy()
                    report_best(best_val, best_x)
                    no_improve = 0
                else:
                    no_improve += 1

                old_mean = mean.copy()
                mean = X[:, :mu] @ weights

                zmean = np.linalg.solve(B @ D, mean - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * zmean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * generation)) < (1.4 + 2 / (dim + 1))) * 1.0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma

                artmp = (X[:, :mu] - old_mean[:, np.newaxis]) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp @ np.diag(weights) @ artmp.T)
                C = (C + C.T) / 2

                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

                if sigma < 1e-12 or no_improve >= 10 + 30 * dim / lam:
                    break

            restart += 1

        return best_val, best_x