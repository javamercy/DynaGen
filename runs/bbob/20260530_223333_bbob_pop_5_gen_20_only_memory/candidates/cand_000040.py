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

        # Latin hypercube sampling
        n_init = min(10, budget // 10)
        if n_init < 2:
            n_init = 2
        init_points = np.zeros((n_init, dim))
        for j in range(dim):
            order = np.random.permutation(n_init)
            init_points[:, j] = (order + np.random.uniform(size=n_init)) / n_init
        init_points = lb + init_points * (ub - lb)

        best_val = np.inf
        best_x = np.zeros(dim)
        evals = 0
        initial_candidates = []
        for i in range(n_init):
            x = init_points[i]
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            initial_candidates.append((val, x.copy()))
        initial_candidates.sort(key=lambda t: t[0])

        # CMA-ES parameters: base lambda
        base_lam = int(4 + 3 * np.log(dim))
        base_lam = max(base_lam, 2)
        lam = base_lam
        use_random_restart = False
        restart_from_best_count = 0
        which_candidate = 1

        while evals < budget:
            if use_random_restart:
                mean = np.random.uniform(lb, ub, size=dim)
                sigma = 0.2 * np.mean(ub - lb)
                use_random_restart = False
                restart_from_best_count = 0
                lam = base_lam  # reset lambda after random restart
            else:
                if restart_from_best_count < 3:
                    mean = best_x.copy()
                    sigma = 0.1 * np.mean(ub - lb)
                    restart_from_best_count += 1
                else:
                    if which_candidate < len(initial_candidates):
                        mean = initial_candidates[which_candidate][1].copy()
                        which_candidate += 1
                    else:
                        mean = best_x.copy()
                    sigma = 0.2 * np.mean(ub - lb)
                    restart_from_best_count = 0
                # Increase lambda after each restart (except after random restart reset)
                lam = int(lam * 1.5)
                if lam > budget - evals:
                    lam = max(2, (budget - evals) // 2)

            # Adapt lambda to remaining budget
            remaining = budget - evals
            lam_run = min(lam, remaining)
            if lam_run < 2:
                break
            mu = lam_run // 2
            if mu < 1:
                mu = 1
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(dim + 1)) - 1) + cs

            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            generation = 0
            no_improve_count = 0
            best_before_run = best_val

            while evals + lam_run <= budget:
                generation += 1
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.diag(np.sqrt(eigvals))
                B = eigvecs
                Z = np.random.randn(dim, lam_run)
                X = mean[:, np.newaxis] + sigma * (B @ D @ Z)
                X = np.clip(X, lb[:, np.newaxis], ub[:, np.newaxis])

                F = np.zeros(lam_run)
                for i in range(lam_run):
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
                    restart_from_best_count = 0
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

                if sigma < 1e-12 or no_improve_count >= max(10, 30 * dim // lam_run):
                    break

            if best_val >= best_before_run - 1e-12:
                use_random_restart = True

        return best_val, best_x