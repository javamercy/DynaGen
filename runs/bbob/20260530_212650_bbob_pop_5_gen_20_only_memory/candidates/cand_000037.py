import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(lb, ub, size=dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        report_best(best_f, best_x)
        calls = 1

        # Step size initialization
        range_scale = np.mean(ub - lb)
        sigma = 0.1 * range_scale  # smaller initial step size
        window = 5  # smaller adaptation window
        successes = 0
        c_inc = 1.2
        c_dec = 0.7  # more aggressive decrease

        while calls < budget:
            # Generate offspring
            y = x + sigma * rng.randn(dim)
            y = np.clip(y, lb, ub)
            if calls >= budget:
                break
            fy = func(y)
            calls += 1
            if fy < fx:
                x = y.copy()
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    report_best(best_f, best_x)
                successes += 1

            # Step size adaptation
            if (calls % window) == 0 and calls > 0:
                rate = successes / window
                if rate > 0.2:
                    sigma *= c_inc
                else:
                    sigma *= c_dec
                successes = 0

            # Restart when sigma too small or budget nearly exhausted
            if sigma < 1e-12 * range_scale or calls >= budget - 5:
                # Restart around best with small perturbation
                x = best_x.copy() + 0.05 * range_scale * rng.randn(dim)
                x = np.clip(x, lb, ub)
                if calls < budget:
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        report_best(best_f, best_x)
                sigma = 0.05 * range_scale
                successes = 0

        return best_f, best_x