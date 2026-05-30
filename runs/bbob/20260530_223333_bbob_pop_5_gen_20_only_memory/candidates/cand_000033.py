import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        best_val = np.inf
        best_x = None
        evals_total = 0
        lam = int(4 + 3 * np.log(self.dim))
        while evals_total < self.budget:
            mean = self.lb + (self.ub - self.lb) * np.random.uniform(0, 1, size=self.dim)
            sigma = 0.5
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            val = func(mean)
            evals_total += 1
            if val < best_val:
                best_val = val
                best_x = mean.copy()
                report_best(best_val, best_x)
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            cc = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
            c1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((self.dim + 2)**2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(self.dim + 1)) - 1) + cs
            while evals_total + lam <= self.budget:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.diag(np.sqrt(eigvals))
                B = eigvecs
                Z = np.random.randn(self.dim, lam)
                X = mean[:, np.newaxis] + sigma * (B @ D @ Z)
                X = np.clip(X, self.lb[:, np.newaxis], self.ub[:, np.newaxis])
                F = np.zeros(lam)
                for i in range(lam):
                    F[i] = func(X[:, i])
                    evals_total += 1
                idx = np.argsort(F)
                F = F[idx]
                X = X[:, idx]
                if F[0] < best_val:
                    best_val = F[0]
                    best_x = X[:, 0].copy()
                    report_best(best_val, best_x)
                old_mean = mean.copy()
                mean = X[:, :mu] @ weights
                zmean = np.linalg.solve(B @ D, mean - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * zmean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(evals_total/lam))) < (1.4 + 2/(self.dim+1))) * 1.0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma
                artmp = (X[:, :mu] - old_mean[:, np.newaxis]) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp @ np.diag(weights) @ artmp.T)
                C = (C + C.T) / 2
                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(self.dim) - 1))
                if sigma < 1e-12:
                    break
            lam = int(lam * 2)
            if lam > self.budget - evals_total:
                break
        return best_val, best_x