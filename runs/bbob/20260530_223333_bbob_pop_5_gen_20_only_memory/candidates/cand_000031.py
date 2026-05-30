import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub

        # initial Latin hypercube sampling (LHS)
        def lhs(n, d, lb, ub):
            samples = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                samples[:, j] = (perm + np.random.uniform(size=n)) / n
            return lb + samples * (ub - lb)

        n_init = min(5, budget // 3)
        init_pts = lhs(n_init, dim, lb, ub)
        best_val = np.inf
        best_x = np.zeros(dim)
        evals = 0
        for i in range(n_init):
            x = init_pts[i]
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # initial CMA-ES population size
        lam = int(4 + 3 * np.log(dim))

        # restart loop
        restart_count = 0
        while evals < budget:
            # adapt lambda to remaining budget
            remaining = budget - evals
            if lam > remaining:
                lam = remaining
            if lam < 2:
                break

            # compute weights and other CMA parameters
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1.0/mu_eff) / ((dim + 2)**2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(dim + 1)) - 1) + cs

            # restart from random point (except first restart from best)
            if restart_count == 0:
                mean = best_x.copy()
            else:
                # random point in bounds using Latin hypercube
                mean = lhs(1, dim, lb, ub).flatten()

            sigma = 0.2 * np.mean(ub - lb)
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            generation = 0
            no_improve_count = 0
            while evals + lam <= budget:
                generation += 1
                # eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.diag(np.sqrt(eigvals))
                B = eigvecs

                # sample offspring
                Z = np.random.randn(dim, lam)
                X = mean[:, np.newaxis] + sigma * (B @ D @ Z)
                X = np.clip(X, lb[:, np.newaxis], ub[:, np.newaxis])

                # evaluate
                F = np.zeros(lam)
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
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                old_mean = mean.copy()
                mean = X[:, :mu] @ weights

                zmean = np.linalg.solve(B @ D, mean - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * zmean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*generation)) < (1.4 + 2/(dim+1))) * 1.0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma

                artmp = (X[:, :mu] - old_mean[:, np.newaxis]) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp @ np.diag(weights) @ artmp.T)
                C = (C + C.T) / 2

                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

                # restart conditions
                if sigma < 1e-12 or no_improve_count >= 10 + 30*dim/lam:
                    break

            # update for next restart
            lam = int(lam * 2)
            restart_count += 1

        return best_val, best_x