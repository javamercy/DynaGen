import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        n = self.dim
        rng = self.rng

        # initialize mean and best
        x_mean = rng.uniform(lb, ub, size=n)
        f_mean = func(x_mean)
        x_best = x_mean.copy()
        f_best = f_mean
        calls = 1
        report_best(f_best, x_best)

        # CMA-ES parameters
        sigma = 0.4 * np.mean(ub - lb)
        C = np.eye(n)
        p_c = np.zeros(n)
        p_sigma = np.zeros(n)
        c_c = 2.0 / (n + 2.0)
        c_cov = 2.0 / (n**2 + 6.0)
        c_sigma = 2.0 / (n + 2.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((n-1.0)/(n+1.0)) - 1.0) + c_sigma
        chi_n = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 1.0/(21.0*n**2))

        no_improve = 0

        while calls < self.budget:
            # Cholesky factor
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                A = np.eye(n)

            z = rng.normal(0, 1, size=n)
            y = A @ z
            x = x_mean + sigma * y
            x = np.clip(x, lb, ub)
            f = func(x)
            calls += 1
            if calls > self.budget:
                break

            if f < f_best:
                f_best = f
                x_best = x.copy()
                report_best(f_best, x_best)
                no_improve = 0

            # update mean if improvement
            if f < f_mean:
                x_mean = x.copy()
                f_mean = f
                # update evolution paths (successful step)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * y
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * z
                # update covariance matrix
                C = (1 - c_cov) * C + c_cov * np.outer(p_c, p_c)
                # update step size
                sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            else:
                no_improve += 1
                if no_improve > 50:
                    sigma *= 0.99
                    # reset paths slightly
                    p_c *= 0.9
                    p_sigma *= 0.9

            # restart if sigma too small or stagnation
            if sigma < 1e-12 * np.mean(ub - lb) or no_improve > 200:
                # generate new random mean
                x_new = rng.uniform(lb, ub, size=n)
                f_new = func(x_new)
                calls += 1
                if calls > self.budget:
                    break
                if f_new < f_best:
                    f_best = f_new
                    x_best = x_new.copy()
                    report_best(f_best, x_best)
                x_mean = x_new
                f_mean = f_new
                # reinitialize CMA state
                sigma = 0.4 * np.mean(ub - lb)
                C = np.eye(n)
                p_c = np.zeros(n)
                p_sigma = np.zeros(n)
                no_improve = 0

        return (f_best, x_best)