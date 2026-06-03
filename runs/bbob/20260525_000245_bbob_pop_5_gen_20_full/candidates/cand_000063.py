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

        if budget - calls < 2:
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
        # Base population size
        lambda_base = 4 + int(3 * np.log(n))
        lambda_base = max(lambda_base, 2)

        # Restart counters
        max_restarts = int(np.log2(budget))  #  adaptive
        restart_attempts = 0

        # local refinement parameters
        local_refine_calls = max(1, int(budget * 0.02))

        for restart in range(max_restarts + 1):
            # Restart with possible larger population
            if restart == 0:
                lambda_ = min(budget - calls, lambda_base)
            else:
                # Increase population by a factor
                lambda_ = min(budget - calls, lambda_base * (2 ** restart))
            if lambda_ < 2:
                break
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

            # Initialize or reinitialize
            if restart == 0:
                # Keep current mean and best
                pass
            else:
                # Perturb mean around a random point
                mean = np.random.uniform(lb, ub, dim)
                # Evaluate initial point of restart
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

            # Stagnation detection
            no_improve_iter = 0
            max_no_improve_iter = max(10, int(budget / (lambda_ * 4)))
            best_val_in_run = best_val

            while calls < budget and no_improve_iter < max_no_improve_iter:
                # Determine actual population size
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

                # Update best
                if vals_sorted[0] < best_val:
                    best_val = vals_sorted[0]
                    best_x = samples_sorted[0].copy()
                    report_best(best_val, best_x)
                    no_improve_iter = 0
                else:
                    no_improve_iter += 1

                # Update mean
                old_mean = mean.copy()
                mean = np.dot(w, samples_sorted[:mu])

                # Compute inverse sqrt of C
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T))
                except:
                    invsqrtC = np.eye(n)

                # Update evolution paths
                ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * np.dot(invsqrtC, (mean - old_mean) / sigma)
                norm_ps = np.linalg.norm(ps)
                expected_norm = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
                sigma = sigma * np.exp((c_s / d_s) * (norm_ps / expected_norm - 1))

                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

                # Update covariance matrix (rank-one and rank-mu)
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc)
                # Rank-mu update only if mu > 1
                if mu > 1 and c_mu > 0:
                    artmp = (samples_sorted[:mu] - old_mean) / sigma
                    C += c_mu * np.dot(artmp.T, np.dot(np.diag(w), artmp))
                C = (C + C.T) / 2
                # Numerical safeguard
                if np.linalg.det(C) <= 0:
                    C = np.eye(n)

                # Restart trigger: small sigma or stagnation
                if sigma < 1e-12 * np.mean(ub - lb) or no_improve_iter >= max_no_improve_iter:
                    break

            # After main loop, if budget remains, local refinement
            if calls < budget:
                # Local refinement: sample around best_x with small perturbations
                local_attempts = min(local_refine_calls, budget - calls)
                for _ in range(local_attempts):
                    step = np.random.uniform(-0.01, 0.01, dim) * (ub - lb)
                    x = np.clip(best_x + step, lb, ub)
                    val = func(x)
                    calls += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

        return best_val, best_x