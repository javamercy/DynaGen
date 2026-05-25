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
        n = self.dim
        budget = self.budget

        # Initial random point
        mean = np.random.uniform(lb, ub, n)
        best_x = mean.copy()
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        if budget < 4:
            for _ in range(budget - calls):
                x = np.random.uniform(lb, ub, n)
                val = func(x)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x
                    report_best(best_val, best_x)
            return best_val, best_x

        # CMA-ES parameters (base)
        lambda_base = 4 + int(3 * np.log(n))
        lambda_ = min(budget - calls, lambda_base)
        lambda_ = max(2, lambda_)
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        c_s = (mu_eff + 2) / (n + mu_eff + 5)
        d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)

        # Initialize state
        sigma = 0.2 * np.mean(ub - lb)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        mean = np.random.uniform(lb, ub, n)  # reinitialize for restart later
        # For restart detection
        no_improve_count = 0
        last_best = best_val
        generation = 0
        restart_factor = 1.0

        while calls < budget:
            # Check for restart conditions
            if sigma < 1e-10 or no_improve_count >= 20:
                # Restart: reinitialize mean, increase population, reset paths, reduce sigma
                restart_factor *= 1.5
                lambda_ = min(budget - calls, int(restart_factor * lambda_base))
                lambda_ = max(2, lambda_)
                mu = lambda_ // 2
                if mu < 1:
                    mu = 1
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mu_eff = 1 / np.sum(w ** 2)
                c_s = (mu_eff + 2) / (n + mu_eff + 5)
                d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
                c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
                c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
                mean = np.random.uniform(lb, ub, n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                sigma = 0.2 * np.mean(ub - lb)
                C = np.eye(n)
                no_improve_count = 0
                last_best = best_val
                generation = 0
                # Evaluate new mean
                val = func(mean)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = mean.copy()
                    report_best(best_val, best_x)
                continue

            # Determine actual population size
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
                samples = mean + sigma * np.random.randn(lambda_actual, n) * np.sqrt(np.diag(C))
            samples = np.clip(samples, lb, ub)

            # Evaluate
            vals = np.array([func(s) for s in samples])
            calls += lambda_actual

            # Sort
            idx = np.argsort(vals)
            vals_sorted = vals[idx]
            samples_sorted = samples[idx]

            # Update best
            if vals_sorted[0] < best_val:
                best_val = vals_sorted[0]
                best_x = samples_sorted[0]
                report_best(best_val, best_x)
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(w, samples_sorted[:mu])

            # Compute invsqrtC
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T))
            except:
                invsqrtC = np.eye(n)

            # Update evolution paths
            ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * np.dot(invsqrtC, (mean - old_mean) / sigma)
            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma = sigma * np.exp((c_s / d_s) * (norm_ps / expected_norm - 1))

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

            # Update covariance (rank-one)
            C = (1 - c_1) * C + c_1 * np.outer(pc, pc)
            C = (C + C.T) / 2

            # Safeguard
            if np.linalg.det(C) <= 0:
                C = np.eye(n)

            generation += 1

        return best_val, best_x