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

        # Helper variables
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

        # Initial random point
        x0 = rng.uniform(lb, ub, size=dim)
        evaluate(x0)
        if count >= budget:
            return best_f, best_x

        # ---------- CMA-ES global search ----------
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        sigma = 0.2 * np.mean(domain_range)
        xmean = x0.copy()
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        invsqrtC = np.eye(dim)
        eigen_eval = 0

        max_cma_calls = int(budget * 0.4)  # allocate 40% budget to CMA
        while count + lam <= budget and count < max_cma_calls:
            arx = []
            arf = []
            for _ in range(lam):
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

            if sigma < 1e-8 * np.mean(domain_range):
                break

        # ---------- Local refinement ----------
        if best_x is None:
            # fallback: uniform random
            while count < budget:
                x = rng.uniform(lb, ub, size=dim)
                evaluate(x)
            return best_f, best_x

        # Ensure eigenvectors are fresh
        C = np.triu(C) + np.triu(C, 1).T
        D, B = np.linalg.eigh(C)
        D = np.abs(D)
        D = np.maximum(D, 1e-30)
        D = np.sqrt(D)

        # Pattern search parameters
        step_size_init = 0.1 * np.mean(domain_range) * D  # per direction
        step_size = step_size_init.copy()
        factor = 1.0

        # Local search: inner loop alternates pattern and random
        candidates_per_cycle = 2 * dim  # pattern: two steps per direction
        random_fraction = 0.3  # 30% of remaining budget for random perturbations
        remaining = budget - count
        random_calls = int(remaining * random_fraction)
        pattern_calls = remaining - random_calls

        while count < budget and pattern_calls > 0:
            improved = False
            for i in range(dim):
                if pattern_calls <= 0:
                    break
                direction = B[:, i]
                step = factor * step_size[i]
                # positive step
                x_new = best_x + step * direction
                x_new = np.clip(x_new, lb, ub)
                f_new = evaluate(x_new)
                pattern_calls -= 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    improved = True
                    break
                if count >= budget:
                    break
                # negative step
                x_new = best_x - step * direction
                x_new = np.clip(x_new, lb, ub)
                f_new = evaluate(x_new)
                pattern_calls -= 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    improved = True
                    break
            if not improved:
                factor *= 0.5  # reduce step size
            else:
                factor = min(1.0, factor * 1.2)  # increase slightly on success
            if factor < 1e-10:
                break

        # Random perturbations (exploration)
        while count < budget and random_calls > 0:
            local_sigma = 0.05 * np.mean(domain_range)
            x_new = best_x + rng.normal(0, local_sigma, size=dim)
            x_new = np.clip(x_new, lb, ub)
            evaluate(x_new)
            random_calls -= 1

        # If any budget left, fill with uniform random (should not happen often)
        while count < budget:
            x_new = rng.uniform(lb, ub, size=dim)
            evaluate(x_new)

        return best_f, best_x