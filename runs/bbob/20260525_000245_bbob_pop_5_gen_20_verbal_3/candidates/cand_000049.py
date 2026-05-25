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

        # Initial feasible point
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

        # CMA-ES parameters
        lambda_ = min(budget - calls, 4 + int(3 * np.log(n)))
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
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        if mu == 1:
            c_mu = 0.0

        # Restart parameters
        max_restarts = 3
        restart_count = 0
        no_improve_thresh = max(1, int(lambda_ * 3))
        mean_range = np.mean(ub - lb)

        while calls < budget and restart_count < max_restarts:
            # Initialize state for current restart
            if restart_count > 0:
                mean = np.random.uniform(lb, ub, n)
                sigma = 0.5 * mean_range
            else:
                sigma = 0.2 * mean_range
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            no_improve_count = 0
            best_val_restart = best_val

            # Main loop for this restart
            while calls < budget:
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

                # Update best overall
                improved = False
                if vals_sorted[0] < best_val:
                    best_val = vals_sorted[0]
                    best_x = samples_sorted[0]
                    report_best(best_val, best_x)
                    no_improve_count = 0
                    best_val_restart = best_val
                    improved = True
                else:
                    no_improve_count += 1

                # Update mean
                old_mean = mean.copy()
                mean = np.dot(w, samples_sorted[:mu])

                # Inverse sqrt C
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
                sigma = max(sigma, 1e-12 * mean_range)

                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

                # Covariance update
                diffs = (samples_sorted[:mu] - old_mean) / sigma
                C_mu = np.zeros((n, n))
                for i in range(mu):
                    C_mu += w[i] * np.outer(diffs[i], diffs[i])
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * C_mu
                C = (C + C.T) / 2
                if np.linalg.det(C) <= 0:
                    C = np.eye(n)

                # Local refinement if stagnation
                if no_improve_count >= 2 and not improved:
                    local_budget = min(5, budget - calls)
                    if local_budget > 0:
                        local_scale = sigma * 0.5
                        for _ in range(local_budget):
                            x = best_x + local_scale * np.random.randn(n)
                            x = np.clip(x, lb, ub)
                            val = func(x)
                            calls += 1
                            if val < best_val:
                                best_val = val
                                best_x = x
                                report_best(best_val, best_x)
                                no_improve_count = 0
                                break

                # Restart condition
                if no_improve_count >= no_improve_thresh or sigma < 1e-10 * mean_range:
                    break

            restart_count += 1

        return best_val, best_x