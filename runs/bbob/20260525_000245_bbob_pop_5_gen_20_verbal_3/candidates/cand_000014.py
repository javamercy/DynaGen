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

        # CMA-ES parameters
        n = dim
        lambda_ = min(budget - calls, 4 + int(3 * np.log(n)))
        lambda_ = max(2, lambda_)
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)

        # Step size control
        c_s = (mu_eff + 2) / (n + mu_eff + 5)
        d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
        # Covariance update
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        if mu == 1:
            c_mu = 0.0

        # Initialize state
        sigma = 0.2 * np.mean(ub - lb)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Main loop
        while calls < budget:
            # Determine actual population size for this iteration
            if calls + lambda_ > budget:
                lambda_actual = budget - calls
            else:
                lambda_actual = lambda_
            if lambda_actual < 1:
                break

            # Sample points
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
            if vals_sorted[0] < best_val:
                best_val = vals_sorted[0]
                best_x = samples_sorted[0]
                report_best(best_val, best_x)

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(w, samples_sorted[:mu])

            # Compute inverse square root of C (for step-size adaptation)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T))
            except:
                invsqrtC = np.eye(n)

            # Update evolution paths
            ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * np.dot(invsqrtC, (mean - old_mean) / sigma)

            # Update step size
            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma = sigma * np.exp((c_s / d_s) * (norm_ps / expected_norm - 1))

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

            # Update covariance matrix (rank-one and rank-mu)
            # Rank-mu term: weighted sum of outer products of (x_i - old_mean)/sigma
            diffs = (samples_sorted[:mu] - old_mean) / sigma
            C_mu = np.zeros((n, n))
            for i in range(mu):
                C_mu += w[i] * np.outer(diffs[i], diffs[i])
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * C_mu
            C = (C + C.T) / 2

            # Numerical safeguard
            if np.linalg.det(C) <= 0:
                C = np.eye(n)

        return best_val, best_x