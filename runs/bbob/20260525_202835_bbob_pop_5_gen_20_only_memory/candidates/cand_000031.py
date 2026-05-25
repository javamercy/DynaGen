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
        dim = self.dim
        # Initial point
        best_x = self.rng.uniform(lb, ub)
        best_val = func(best_x)
        report_best(best_val, best_x)
        evaluations = 1
        # Initial step size: 0.2 * range per dimension
        sigma = 0.2 * (ub - lb)
        # Success rate tracking
        window = 10
        successes = 0
        total = 0
        # Main loop
        while evaluations < self.budget:
            # Generate offspring
            noise = self.rng.normal(0, 1, dim)
            x_new = np.clip(best_x + sigma * noise, lb, ub)
            val_new = func(x_new)
            evaluations += 1
            # Selection
            if val_new < best_val:
                best_val = val_new
                best_x = x_new.copy()
                report_best(best_val, best_x)
                successes += 1
            total += 1
            # Update step size based on success rate in window
            if total >= window:
                success_rate = successes / window
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                successes = 0
                total = 0
        return best_val, best_x