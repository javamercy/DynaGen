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

        if budget <= 2:
            for _ in range(budget - calls):
                x = np.random.uniform(lb, ub, n)
                val = func(x)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x
                    report_best(best_val, best_x)
            return best_val, best_x

        # Reduced population for exploitation
        lambda_ = min(budget - calls, max(2, int(2 + 2 * np.log(n))))
        lambda_ = max(2, lambda_)
        mu = max(1, lambda_ // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Adaptation rates (increased for faster convergence)
        c_s = (mu_eff + 2) / (n + mu_eff + 5)
        d_s = 1 + 2 * max(0, np.sqrt((mu_eff-1)/(n+1)) - 1) + c_s
        c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        c_1 = 2 / ((n+1.3)**2 + mu_eff)  # rank-one only
        c_1 = min(c_1, 0.1)  # cap for stability

        sigma = 0.1 * np.mean(ub - lb)  # smaller initial sigma
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        stall_counter = 0
        max_stall = max(10, n)

        while calls < budget:
            if calls + lambda_ > budget:
                lambda_actual = budget - calls
            else:
                lambda_actual = lambda_
            if lambda_actual < 1:
                break

            # Sample
            try:
                samples = np.random.multivariate_normal(mean, sigma**2 * C, size=lambda_actual)
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
                stall_counter = 0
            else:
                stall_counter += 1

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(w, samples_sorted[:mu])

            # Compute invsqrtC
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0/np.sqrt(eigvals)), eigvecs.T))
            except:
                invsqrtC = np.eye(n)

            ps = (1 - c_s) * ps + np.sqrt(c_s*(2-c_s)*mu_eff) * np.dot(invsqrtC, (mean - old_mean)/sigma)
            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma *= np.exp((c_s/d_s) * (norm_ps/expected_norm - 1))
            sigma = max(sigma, 1e-10)

            pc = (1 - c_c) * pc + np.sqrt(c_c*(2-c_c)*mu_eff) * (mean - old_mean)/sigma
            C = (1 - c_1) * C + c_1 * np.outer(pc, pc)
            C = (C + C.T) / 2
            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) <= 0:
                C += np.eye(n) * (1e-15 - np.min(eigvals))

            # Restart if stalled
            if stall_counter >= max_stall:
                # Reset mean to best point with small perturbation
                mean = best_x + 0.05 * sigma * np.random.randn(n)
                mean = np.clip(mean, lb, ub)
                sigma = 0.1 * np.mean(ub - lb)
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                stall_counter = 0

        return best_val, best_x