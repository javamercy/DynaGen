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
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        # Initial point
        best_x = rng.uniform(lb, ub)
        best_f = func(best_x)
        calls = 1
        report_best(best_f, best_x)

        # Step sizes: initial 10% of domain range per dimension
        delta = 0.1 * (ub - lb)

        # Main loop
        while calls < budget:
            improved = False
            for i in range(dim):
                if calls >= budget:
                    break
                # Positive perturbation
                cand = best_x.copy()
                cand[i] += delta[i]
                cand = np.clip(cand, lb, ub)
                f = func(cand)
                calls += 1
                if f < best_f:
                    best_x = cand
                    best_f = f
                    report_best(best_f, best_x)
                    delta[i] *= 2.0
                    improved = True
                    continue  # skip negative
                # Negative perturbation
                cand = best_x.copy()
                cand[i] -= delta[i]
                cand = np.clip(cand, lb, ub)
                f = func(cand)
                calls += 1
                if f < best_f:
                    best_x = cand
                    best_f = f
                    report_best(best_f, best_x)
                    delta[i] *= 2.0
                    improved = True
                else:
                    delta[i] *= 0.5
            # If no improvement after full cycle, optionally shrink all step sizes (already done per coordinate)
            # No extra action needed

        return best_f, best_x