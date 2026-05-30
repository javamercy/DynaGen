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

        # initial point
        x = rng.uniform(lb, ub, size=dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        report_best(best_f, best_x)
        calls = 1

        # step size initialization
        range_mean = np.mean(ub - lb)
        sigma = 0.2 * range_mean
        success_counter = 0
        eval_window = 10

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
            # step size adaptation
            if (calls % eval_window) == 0:
                success_rate = success_counter / eval_window
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                success_counter = 0
            # restart condition
            if sigma < 1e-8 * range_mean or calls >= budget - 5:
                x = rng.uniform(lb, ub, size=dim)
                # evaluate new point immediately if budget left
                if calls < budget:
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        report_best(best_f, best_x)
                sigma = 0.2 * range_mean
                success_counter = 0

        return best_f, best_x