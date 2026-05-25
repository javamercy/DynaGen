import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        if budget == 0:
            return np.inf, np.zeros(dim)

        # Initial point
        x = lb + rng.rand(dim) * (ub - lb)
        best_x = x.copy()
        best_val = func(x)
        evals = 1
        report_best(best_val, best_x)

        sigma = 0.2 * np.mean(ub - lb)
        success_counter = 0
        generation = 0
        learn_rate = 0.817  # 1/(sqrt(dim)+1) approx, adaption rate

        while evals < budget:
            # Generate offspring
            step = sigma * rng.randn(dim)
            candidate = x + step
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1

            if val < best_val:
                best_val = val
                best_x = candidate.copy()
                x = candidate.copy()
                success_counter += 1
                report_best(best_val, best_x)
            else:
                # Reject: keep parent
                pass

            generation += 1
            # Adapt sigma every 1/dim generations (approx)
            if generation >= dim:
                success_rate = success_counter / generation
                target = 0.2
                if success_rate > target:
                    sigma *= 1.0 / 0.85  # increase
                else:
                    sigma *= 0.85  # decrease
                # Reset counters
                success_counter = 0
                generation = 0

        return best_val, best_x