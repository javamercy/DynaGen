import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        range_mean = np.mean(ub - lb)

        # initial point
        x = rng.uniform(lb, ub, size=dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        report_best(best_f, best_x)
        calls = 1

        # step size parameters: smaller initial sigma for exploitation
        sigma = 0.1 * range_mean
        success_counter = 0
        eval_window = 10
        restart_threshold = 1e-10 * range_mean

        while calls < budget:
            # generate offspring
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
                success_counter += 1

            # step size adaptation (1/5 rule with conservative factors)
            if (calls % eval_window) == 0:
                success_rate = success_counter / eval_window
                if success_rate > 0.2:
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                success_counter = 0

            # restart from best point with small initial sigma
            if sigma < restart_threshold or calls >= budget - 5:
                # reset to best point with small perturbation
                x = best_x + 0.01 * sigma * rng.randn(dim)
                x = np.clip(x, lb, ub)
                sigma = 0.05 * range_mean
                if calls < budget:
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        report_best(best_f, best_x)
                success_counter = 0

        return best_f, best_x