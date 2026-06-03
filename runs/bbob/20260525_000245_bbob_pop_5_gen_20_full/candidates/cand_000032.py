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

        # Initialize and evaluate first point
        mean = np.random.uniform(lb, ub, n)
        best_x = mean.copy()
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        # CMA-ES parameters
        lambda_ = min(budget - calls, 4 + int(3 * np.log(n)))
        lambda_ = max(2, lambda_)
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)

        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        if mu == 1:
            cmu = 0.0

        sigma = 0.25 * np.mean(ub - lb)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Restart tracking
        no_improve_iter = 0
        max_stall = max(5, int(budget / (lambda_ * 2)))
        # Population increase factor
        lambda_factor = 1.5
        current_lambda = lambda_

        while calls < budget:
            # Determine actual population size
            if calls + current_lambda > budget:
                lambda_actual = budget - calls
            else:
                lambda_actual = current_lambda
            if lambda_actual < 1:
                break

            # Sample points
            try:
                samples = np.random.multivariate_normal(mean, sigma ** 2 * C, size=lambda_actual)
            except:
                samples = mean + sigma * np.random.randn(lambda_actual, n) * np.sqrt(np.diag(C))
            samples = np.clip(samples, lb, ub)

            # Evaluate
            vals = np.empty(lambda_actual)
            for i, s in enumerate(samples):
                vals[i] = func(s)
            calls += lambda_actual

            idx = np.argsort(vals)
            vals_sorted = vals[idx]
            samples_sorted = samples[idx]

            # Update best
            if vals_sorted[0] < best_val:
                best_val = vals_sorted[0]
                best_x = samples_sorted[0]
                report_best(best_val, best_x)
                no_improve_iter = 0
            else:
                no_improve_iter += 1

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(w, samples_sorted[:mu])

            # Compute inverse sqrt C
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T))
            except:
                invsqrtC = np.eye(n)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * np.dot(invsqrtC, (mean - old_mean) / sigma)
            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma *= np.exp((cs / ds) * (norm_ps / expected_norm - 1))
            sigma = max(sigma, 1e-12 * np.mean(ub - lb))  # avoid zero

            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma

            # Update covariance matrix
            diffs = (samples_sorted[:mu] - old_mean) / sigma
            Cmu = np.zeros((n, n))
            for i in range(mu):
                Cmu += w[i] * np.outer(diffs[i], diffs[i])
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * Cmu
            C = (C + C.T) / 2
            # Regularize for positive definiteness
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(n)

            # Restart condition
            restart = False
            if sigma < 1e-12 * np.mean(ub - lb):
                restart = True
            if no_improve_iter >= max_stall:
                restart = True

            if restart and calls < budget:
                # Increase population size for diversity
                current_lambda = min(int(current_lambda * lambda_factor), (budget - calls) // 2)
                current_lambda = max(current_lambda, 2)
                mu = current_lambda // 2
                if mu < 1:
                    mu = 1
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mu_eff = 1 / np.sum(w ** 2)
                # Update CMA parameters for new lambda
                cs = (mu_eff + 2) / (n + mu_eff + 5)
                ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
                cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
                c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
                cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
                if mu == 1:
                    cmu = 0.0
                # Reinitialize state around best point with larger sigma
                sigma = 0.3 * np.mean(ub - lb)
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                mean = best_x + 0.1 * np.random.uniform(-1, 1, n) * (ub - lb)
                mean = np.clip(mean, lb, ub)
                no_improve_iter = 0
                max_stall = max(5, int(budget / (current_lambda * 2)))

        return best_val, best_x

# Helper function report_best is expected to be defined by the evaluator