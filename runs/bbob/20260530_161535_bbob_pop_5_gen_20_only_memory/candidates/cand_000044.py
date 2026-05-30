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
        total_budget = self.budget
        rng = self.rng

        best_f = np.inf
        best_x = None
        total_count = 0

        # Number of restarts: at least 3, up to 10
        num_restarts = max(3, min(10, total_budget // (4 * dim)))
        budget_per_restart = total_budget // num_restarts

        def evaluate(x):
            nonlocal total_count, best_f, best_x
            x = np.clip(x, lb, ub)
            f = func(x)
            total_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(f, best_x)
            return f

        for restart in range(num_restarts):
            if total_count >= total_budget:
                break
            remaining = total_budget - total_count
            if restart < num_restarts - 1:
                max_evals_for_run = min(budget_per_restart, remaining)
            else:
                max_evals_for_run = remaining
            if max_evals_for_run < 1:
                break

            # Initial random point
            xmean = rng.uniform(lb, ub, size=dim)
            xmean = np.clip(xmean, lb, ub)
            fmean = evaluate(xmean)
            if total_count >= total_budget:
                break

            # CMA-ES parameters (more exploratory)
            lam = 10 + int(5 * np.log(dim))
            lam = min(lam, max_evals_for_run)
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2) ** 2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

            sigma = 0.7 * np.mean(domain_range)  # larger initial step
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigen_eval = 0

            local_count = 1
            gen_no_improve = 0
            best_f_local = fmean

            while local_count + lam <= max_evals_for_run and total_count < total_budget:
                arx = []
                arf = []
                for k in range(lam):
                    z = rng.normal(0, 1, dim)
                    y = B @ (D * z)
                    x = xmean + sigma * y
                    x = np.clip(x, lb, ub)
                    arx.append(x)
                    f = evaluate(x)
                    local_count += 1
                    arf.append(f)
                    if total_count >= total_budget:
                        break
                if total_count >= total_budget:
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

                if local_count - eigen_eval > dim:
                    eigen_eval = local_count
                    C = np.triu(C) + np.triu(C, 1).T
                    D, B = np.linalg.eigh(C)
                    D = np.abs(D)
                    D = np.maximum(D, 1e-30)
                    D = np.sqrt(D)
                    invsqrtC = B @ np.diag(1/D) @ B.T

                # Stagnation detection and diversification
                if arf[idx[0]] < best_f_local:
                    best_f_local = arf[idx[0]]
                    gen_no_improve = 0
                else:
                    gen_no_improve += 1

                # Early restart if sigma too small or stagnation
                if sigma < 1e-6 * np.mean(domain_range) or gen_no_improve >= 5:
                    break

            # End of restart loop

        return best_f, best_x