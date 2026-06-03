import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.stagnation_limit = max(1, budget // 10)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        rng = self.rng
        budget = self.budget

        # Initial scale as 20% of domain range per dimension
        initial_scale = 0.2 * (ub - lb)
        scale = initial_scale.copy()

        # Initial random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        report_best(best_val, best_x)
        evals = 1
        no_improve = 0

        while evals < budget:
            # Generate candidate by perturbing best
            candidate = best_x + rng.normal(0, scale, size=dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1

            if val < best_val:
                best_val = val
                best_x = candidate
                report_best(best_val, best_x)
                no_improve = 0
                # Shrink scale to exploit
                scale = np.maximum(scale * 0.5, 1e-8 * (ub - lb))
            else:
                no_improve += 1
                if no_improve >= self.stagnation_limit:
                    if evals >= budget:
                        break
                    # Restart: evaluate a new random point
                    new_x = rng.uniform(lb, ub)
                    new_val = func(new_x)
                    evals += 1
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x
                        report_best(best_val, best_x)
                    no_improve = 0
                    scale = initial_scale.copy()
                else:
                    # Gradually expand scale when not improving
                    if no_improve % max(1, self.stagnation_limit // 4) == 0:
                        scale = np.minimum(scale * 1.5, ub - lb)

        return best_val, best_x