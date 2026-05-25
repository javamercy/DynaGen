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

        # Base population size
        def get_lambda(calls):
            base = 4 + int(4 * np.log(n))
            if calls < budget / 2:
                return max(2, min(budget - calls, base))
            else:
                return max(2, min(budget - calls, base + int(2 * np.log(n))))

        lambda_ = get_lambda(calls)
        if lambda_ < 2:
            lambda_ = 2
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

        sigma = 0.3 * np.mean(ub - lb)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        no_improve_iter = 0
        max_no_improve = max(3, int(budget / (lambda_ * 5)))

        while calls < budget:
            lambda_ = get_lambda(calls)
            if lambda_ > budget - calls:
                lambda_ = budget - calls
            if lambda_ < 1:
                break

            try:
                samples = np.random.multivariate_normal(mean, sigma ** 2 * C, size=lambda_)
            except:
                samples = mean + sigma * np.random.randn(lambda_, n) * np.sqrt(np.diag(C))
            samples = np.clip(samples, lb, ub)

            vals = np.array([func(s) for s in samples])
            calls += lambda_

            idx = np.argsort(vals)
            vals_sorted = vals[idx]
            samples_sorted = samples[idx]

            if vals_sorted[0] < best_val:
                best_val = vals_sorted[0]
                best_x = samples_sorted[0]
                report_best(best_val, best_x)
                no_improve_iter = 0
            else:
                no_improve_iter += 1

            old_mean = mean.copy()
            mean = np.dot(w, samples_sorted[:mu])

            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T))
            except:
                invsqrtC = np.eye(n)

            ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * np.dot(invsqrtC, (mean - old_mean) / sigma)
            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma *= np.exp((c_s / d_s) * (norm_ps / expected_norm - 1))

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

            diffs = (samples_sorted[:mu] - old_mean) / sigma
            C_mu = np.zeros((n, n))
            for i in range(mu):
                C_mu += w[i] * np.outer(diffs[i], diffs[i])
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * C_mu
            C = (C + C.T) / 2

            # Restart conditions (no condition number check)
            restart = False
            if sigma < 1e-12 * np.mean(ub - lb):
                restart = True
            if no_improve_iter >= max_no_improve:
                restart = True

            if restart and calls < budget:
                mean = best_x + 0.5 * np.random.randn(n) * (ub - lb) / np.sqrt(n)
                mean = np.clip(mean, lb, ub)
                sigma = 0.4 * np.mean(ub - lb)
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                no_improve_iter = 0
                max_no_improve = max(3, int(budget / (get_lambda(calls) * 5)))

        # Random perturbation local search (at most 5 evaluations)
        remaining = budget - calls
        if remaining > 0:
            perturb_scale = 0.1 * np.mean(ub - lb)
            for _ in range(min(remaining, 5)):
                candidate = best_x + perturb_scale * np.random.randn(n)
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate
                    report_best(best_val, best_x)

        return best_val, best_x