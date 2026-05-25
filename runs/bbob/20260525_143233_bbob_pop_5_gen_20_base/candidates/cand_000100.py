import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub
        if budget <= 0:
            best_x = np.clip((lb + ub) / 2.0, lb, ub)
            best_val = func(best_x)
            return best_val, best_x
        # Initial parent
        best_x = rng.uniform(lb, ub, dim)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)
        if budget == 1:
            return best_val, best_x
        # Step size initialization
        sigma = 0.2 * (ub - lb).mean()
        min_sigma = 1e-10
        # Success rule parameters
        target_success_rate = 1.0 / 5.0
        adaptation_factor = 0.85
        while evals < budget:
            # Generate offspring
            offspring = best_x + sigma * rng.randn(dim)
            offspring = np.clip(offspring, lb, ub)
            val = func(offspring)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = offspring.copy()
                report_best(best_val, best_x)
                # Increase step size (success)
                sigma = sigma / adaptation_factor
            else:
                # Decrease step size
                sigma = sigma * adaptation_factor
            # Enforce minimum step size
            if sigma < min_sigma:
                sigma = min_sigma
        return best_val, best_x