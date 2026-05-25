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

        # Initial feasible point
        best_x = np.random.uniform(lb, ub, dim)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        # For very small budget, random search
        if budget < 6:
            for _ in range(budget - calls):
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x
                    report_best(best_val, best_x)
            return best_val, best_x

        # Reserve budget for local refinement
        reserve = max(5, int(0.1 * budget))
        main_budget = budget - reserve

        # CMA-ES parameters (more exploitative)
        n = dim
        lambda_ = min(main_budget - calls, 4 + int(2 * np.log(n)))
        lambda_ = max(2, lambda_)
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)

        # Step size control (more aggressive)
        c_s = (mu_eff + 2) / (n + mu_eff + 5)
        d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
        # Covariance update (higher learning rate)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2.0 / (n ** 1.5)  # larger than default
        c_mu = 0

        # Initialize state
        sigma = 0.2 * np.mean(ub - lb)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Main loop
        while calls < main_budget:
            if calls + lambda_ > main_budget:
                lambda_actual = main_budget - calls
            else:
                lambda_actual = lambda_
            if lambda_actual < 1:
                break

            try:
                samples = np.random.multivariate_normal(mean, sigma ** 2 * C, size=lambda_actual)
            except:
                samples = mean + sigma * np.random.randn(lambda_actual, n) * np.sqrt(np.diag(C))
            samples = np.clip(samples, lb, ub)

            vals = np.array([func(s) for s in samples])
            calls += lambda_actual

            idx = np.argsort(vals)
            vals_sorted = vals[idx]
            samples_sorted = samples[idx]

            if vals_sorted[0] < best_val:
                best_val = vals_sorted[0]
                best_x = samples_sorted[0]
                report_best(best_val, best_x)

            old_mean = mean.copy()
            mean = np.dot(w, samples_sorted[:mu])

            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T))
            except:
                invsqrtC = np.eye(n)

            ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * np.dot(invsqrtC, (mean - old_mean) / sigma)

            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma = sigma * np.exp((c_s / d_s) * (norm_ps / expected_norm - 1))

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

            C = (1 - c_1) * C + c_1 * np.outer(pc, pc)
            C = (C + C.T) / 2

            if np.linalg.det(C) <= 0:
                C = np.eye(n)

        # Local random refinement
        sigma_local = 0.1 * np.mean(ub - lb)
        for i in range(reserve):
            step_size = sigma_local * (1 - i / reserve)
            x = best_x + step_size * np.random.randn(dim)
            x = np.clip(x, lb, ub)
            val = func(x)
            calls += 1
            if val < best_val:
                best_val = val
                best_x = x
                report_best(best_val, best_x)

        return best_val, best_x