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

        # Step size initialization (smaller)
        range_mean = np.mean(ub - lb)
        sigma = 0.1 * range_mean
        sigma_min = 1e-12 * range_mean

        # Window for success rate (shorter)
        window = 10
        successes = []

        while calls < budget:
            # Generate one offspring
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
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
            # Restart if sigma too small, but keep best point
            if sigma < sigma_min:
                x = best_x.copy() + 0.05 * range_mean * rng.randn(dim)
                np.clip(x, lb, ub, out=x)
                if calls < budget:
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        report_best(best_f, best_x)
                sigma = 0.1 * range_mean
                successes.clear()

        return best_f, best_x