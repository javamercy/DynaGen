import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        n = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget

        # initial feasible point
        best_x = np.random.uniform(lb, ub, n)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        if budget <= 1:
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

        max_restarts = 3
        restart_count = 0
        no_improve_thresh = max(1, int(lambda_ * 5))
        mean_range = np.mean(ub - lb)

        while calls < budget and restart_count < max_restarts:
            if restart_count == 0:
                mean = best_x.copy()
                sigma = 0.2 * mean_range
            else:
                mean = np.random.uniform(lb, ub, n)
                sigma = 0.5 * mean_range
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            no_improve_count = 0
            best_val_restart = best_val

            while calls < budget:
                if calls + lambda_ > budget:
                    lambda_actual = budget - calls
                else:
                    lambda_actual = lambda_
                if lambda_actual < 1:
                    break

                # Sample
                try:
                    samples = np.random.multivariate_normal(mean, sigma ** 2 * C, size=lambda_actual)
                except np.linalg.LinAlgError:
                    samples = mean + sigma * np.random.randn(lambda_actual, n) * np.sqrt(np.diag(C))
                samples = np.clip(samples, lb, ub)

                # Evaluate
                vals = np.array([func(s) for s in samples])
                calls += lambda_actual

                idx = np.argsort(vals)
                vals_sorted = vals[idx]
                samples_sorted = samples[idx]

                if vals_sorted[0] < best_val:
                    best_val = vals_sorted[0]
                    best_x = samples_sorted[0]
                    report_best(best_val, best_x)
                    no_improve_count = 0
                    best_val_restart = best_val
                else:
                    no_improve_count += 1

                old_mean = mean.copy()
                mean = np.dot(w, samples_sorted[:mu])

                # Inverse sqrt C
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T))
                except np.linalg.LinAlgError:
                    invsqrtC = np.eye(n)

                # Update evolution paths
                ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * np.dot(invsqrtC, (mean - old_mean) / sigma)
                norm_ps = np.linalg.norm(ps)
                expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                sigma = sigma * np.exp((c_s / d_s) * (norm_ps / expected_norm - 1))
                sigma = max(sigma, 1e-12 * mean_range)

                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

                diffs = (samples_sorted[:mu] - old_mean) / sigma
                C_mu = np.zeros((n, n))
                for i in range(mu):
                    C_mu += w[i] * np.outer(diffs[i], diffs[i])
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * C_mu
                C = (C + C.T) / 2
                # Ensure positive definite
                eigvals, _ = np.linalg.eigh(C)
                if np.any(eigvals <= 0):
                    C = np.eye(n)

                if no_improve_count >= no_improve_thresh or sigma < 1e-10 * mean_range:
                    break

            restart_count += 1

        return best_val, best_x