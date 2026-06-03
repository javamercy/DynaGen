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

        if budget <= 1:
            return best_val, best_x

        # CMA-ES parameters
        # Base population size
        lambda_base = 4 + int(3 * np.log(n))
        lambda_ = lambda_base
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

        # Initial state
        sigma = 0.2 * np.mean(ub - lb)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Restart parameters
        max_restarts = 5
        restart_count = 0
        generation_without_improvement = 0
        tol_x = 1e-12 * np.mean(ub - lb)

        # Stagnation detection parameters
        best_val_hist = [best_val]

        while calls < budget:
            # Check budget for at least one generation
            if calls + lambda_ > budget:
                lambda_actual = budget - calls
                if lambda_actual < 1:
                    break
            else:
                lambda_actual = lambda_

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
                generation_without_improvement = 0
            else:
                generation_without_improvement += 1

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(w, samples_sorted[:mu])

            # invsqrtC
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

            # Update covariance
            C = (1 - c_1) * C + c_1 * np.outer(pc, pc)
            C = (C + C.T) / 2
            if np.linalg.det(C) <= 0:
                C = np.eye(n)

            # Restart condition
            restart_trigger = False
            if sigma < tol_x:
                restart_trigger = True
            # Stagnation: no improvement for max(10, 30*n/lambda_) generations
            stagnation_limit = int(max(10, 30 * n / lambda_))
            if generation_without_improvement >= stagnation_limit:
                restart_trigger = True

            if restart_trigger and restart_count < max_restarts and calls < budget:
                # Restart with doubled population size
                restart_count += 1
                lambda_ = min(lambda_ * 2, budget - calls)
                lambda_ = max(2, lambda_)
                mu = lambda_ // 2
                if mu < 1:
                    mu = 1
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mu_eff = 1 / np.sum(w ** 2)
                # Recompute parameters that depend on lambda
                c_s = (mu_eff + 2) / (n + mu_eff + 5)
                d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
                c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
                c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)

                # Reset state
                mean = np.random.uniform(lb, ub, n)
                sigma = 0.2 * np.mean(ub - lb)
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                generation_without_improvement = 0

        return best_val, best_x