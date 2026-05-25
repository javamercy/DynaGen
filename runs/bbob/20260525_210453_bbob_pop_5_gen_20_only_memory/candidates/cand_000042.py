import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial random point
        best_x = np.random.uniform(lb, ub, dim)
        best_f = func(best_x)
        fcalls = 1
        report_best(best_f, best_x)
        # Initial step size as 20% of the average range
        sigma = 0.2 * (ub - lb).mean()
        success_counter = 0
        total_attempts = 0
        while fcalls < budget:
            # Generate offspring
            step = np.random.normal(0, sigma, dim)
            candidate = best_x + step
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            fcalls += 1
            total_attempts += 1
            if val < best_f:
                best_f = val
                best_x = candidate.copy()
                report_best(best_f, best_x)
                success_counter += 1
            # Update sigma every 10 attempts using 1/5 success rule
            if total_attempts % 10 == 0 and total_attempts > 0:
                success_rate = success_counter / 10.0
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                # Reset counters for next window
                success_counter = 0
                total_attempts = 0
            # Keep sigma positive and bounded
            sigma = max(sigma, 1e-12 * (ub - lb).mean())
            sigma = min(sigma, (ub - lb).mean())
        return best_f, best_x