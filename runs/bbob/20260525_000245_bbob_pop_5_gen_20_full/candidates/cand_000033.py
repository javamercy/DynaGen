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
        mean = np.random.uniform(lb, ub, dim)
        best_x = mean.copy()
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        if budget < 4:
            for _ in range(budget - calls):
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x
                    report_best(best_val, best_x)
            return best_val, best_x

        # CMA-ES parameters
        n = dim
        lambda_ = min(budget - calls, 4 + int(3 * np.log(n)))
        lambda_ = max(2, lambda_)
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)

        # Step size control
        c_s = (mu_eff + 2) / (n + mu_eff + 5)
        d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)

        # State
        sigma = 0.2 * np.mean(ub - lb)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Restart tracking
        max_restarts = 2
        restart_count = 0
        stagnation_counter = 0
        prev_best_val = best_val

        while calls < budget:
            # Determine actual lambda
            if calls + lambda_ > budget:
                lambda_actual = budget - calls
            else:
                lambda_actual = lambda_
            if lambda_actual < 1:
                break

            # Sample
            try:
                samples = np.random.multivariate_normal(mean, sigma ** 2 * C, size=lambda_actual)
            except:
                samples = mean + sigma * np.random.randn(lambda_actual, n) * np.sqrt(np.diag(C) + 1e-20)
            samples = np.clip(samples, lb, ub)

            vals = np.array([func(s) for s in samples])
            calls += lambda_actual

            idx = np.argsort(vals)
            vals_sorted = vals[idx]
            samples_sorted = samples[idx]

            # Update best
            if vals_sorted[0] < best_val:
                best_val = vals_sorted[0]
                best_x = samples_sorted[0]
                report_best(best_val, best_x)

            # Restart triggers
            if best_val < prev_best_val:
                stagnation_counter = 0
                prev_best_val = best_val
            else:
                stagnation_counter += 1

            # Check for restart
            if restart_count < max_restarts:
                restart_condition = (sigma < 1e-10) or (stagnation_counter >= 20)
                if restart_condition:
                    # Increase population size
                    lambda_ = min(budget - calls, 2 * lambda_)
                    if lambda_ < 2:
                        lambda_ = 2
                    mu = lambda_ // 2
                    if mu < 1:
                        mu = 1
                    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    w = w / w.sum()
                    mu_eff = 1 / np.sum(w ** 2)
                    # Reset state
                    mean = np.random.uniform(lb, ub, dim)
                    sigma = 0.2 * np.mean(ub - lb)
                    C = np.eye(n)
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    restart_count += 1
                    stagnation_counter = 0
                    continue

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(w, samples_sorted[:mu])

            # Inverse sqrt of C
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

            # Safeguard
            if np.linalg.det(C) <= 0:
                C = np.eye(n)

        return best_val, best_x