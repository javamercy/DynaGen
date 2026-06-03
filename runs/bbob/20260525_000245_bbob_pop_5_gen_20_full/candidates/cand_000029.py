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

        # For very small budget, random search
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

        # CMA-ES parameters (base)
        n = dim
        lambda_base = max(2, min(budget - calls, 4 + int(3 * np.log(n))))
        max_restarts = 3
        restart_count = 0

        # Outer loop for restarts
        while calls < budget and restart_count <= max_restarts:
            # Increase population size with restarts
            lambda_ = int(lambda_base * (1.5 ** restart_count))
            lambda_ = min(lambda_, budget - calls)
            if lambda_ < 2:
                break

            # Initialize CMA-ES state for this restart
            if restart_count > 0:
                # Reinitialize mean randomly (within bounds) and evaluate
                mean = np.random.uniform(lb, ub, dim)
                val = func(mean)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = mean.copy()
                    report_best(best_val, best_x)

            sigma = 0.2 * np.mean(ub - lb)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Counters for stagnation
            stagnation_counter = 0
            stagnation_limit = max(10, n)

            # Inner CMA-ES loop until restart trigger or budget exhausted
            while calls < budget:
                # Determine actual population size for this iteration
                if calls + lambda_ > budget:
                    lambda_actual = budget - calls
                else:
                    lambda_actual = lambda_
                if lambda_actual < 1:
                    break

                # Compute CMA parameters based on current mu_eff
                mu = lambda_actual // 2
                if mu < 1:
                    mu = 1
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mu_eff = 1 / np.sum(w ** 2)

                c_s = (mu_eff + 2) / (n + mu_eff + 5)
                d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
                c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
                c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)

                # Sample
                try:
                    samples = np.random.multivariate_normal(mean, sigma ** 2 * C, size=lambda_actual)
                except:
                    samples = mean + sigma * np.random.randn(lambda_actual, n) * np.sqrt(np.diag(C))
                samples = np.clip(samples, lb, ub)

                # Evaluate
                vals = np.array([func(s) for s in samples])
                calls += lambda_actual

                # Sort by fitness
                idx = np.argsort(vals)
                vals_sorted = vals[idx]
                samples_sorted = samples[idx]

                # Update best solution
                improved = False
                if vals_sorted[0] < best_val:
                    best_val = vals_sorted[0]
                    best_x = samples_sorted[0].copy()
                    report_best(best_val, best_x)
                    improved = True

                # Stagnation update
                if improved:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

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

                # Update covariance matrix (rank-one)
                C = (1 - c_1) * C + c_1 * np.outer(pc, pc)
                C = (C + C.T) / 2

                # Numerical safeguard
                if np.linalg.det(C) <= 0:
                    C = np.eye(n)

                # Restart conditions
                if sigma < 1e-8 * np.mean(ub - lb):
                    restart_count += 1
                    break
                if stagnation_counter >= stagnation_limit:
                    restart_count += 1
                    break

            # End inner loop
            # If we broke due to restart, outer loop continues with increased restart_count and new mean

        return best_val, best_x