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

        # Initial point
        x = rng.uniform(lb, ub, size=dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        report_best(best_f, best_x)
        calls = 1

        # Step size initialization
        range_mean = np.mean(ub - lb)
        sigma = 0.2 * range_mean
        sigma_min = 1e-12 * range_mean

        # Window for success rate
        window = 20
        successes = []

        while calls < budget:
            # Number of offspring to generate this iteration
            lambda_ = min(4, budget - calls)
            if lambda_ == 0:
                break
            for i in range(lambda_):
                if calls >= budget:
                    break
                y = x + sigma * rng.randn(dim)
                np.clip(y, lb, ub, out=y)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        report_best(best_f, best_x)
                    successes.append(True)
                else:
                    successes.append(False)
                # Keep window size
                if len(successes) > window:
                    successes.pop(0)
                # Adapt sigma when window is full
                if len(successes) >= window:
                    success_rate = np.mean(successes[-window:])
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
            # Restart if sigma too small
            if sigma < sigma_min:
                x = rng.uniform(lb, ub, size=dim)
                if calls < budget:
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        report_best(best_f, best_x)
                sigma = 0.2 * range_mean
                successes.clear()

        return best_f, best_x