import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = np.random.RandomState(self.seed)
        dim = self.dim
        budget = self.budget

        # initial point
        x = rng.uniform(lb, ub)
        best_x = x.copy()
        best_val = func(x)
        evals = 1
        report_best(best_val, best_x)

        # step size
        sigma = 0.2 * np.mean(ub - lb)
        # adaptation
        success_target = 1.0 / 5.0
        window = 10
        success_history = []
        inc_factor = 1.2
        dec_factor = 0.8

        while evals < budget:
            # mutate
            mutant = x + sigma * rng.randn(dim)
            mutant = np.clip(mutant, lb, ub)
            val = func(mutant)
            evals += 1

            # update
            if val < best_val:
                best_val = val
                best_x = mutant.copy()
                report_best(best_val, best_x)
                x = mutant.copy()
                success_history.append(True)
            else:
                success_history.append(False)

            # step-size adaptation
            if len(success_history) == window:
                success_rate = sum(success_history) / window
                if success_rate > success_target:
                    sigma *= inc_factor
                elif success_rate < success_target:
                    sigma *= dec_factor
                success_history = []
                # bound sigma
                sigma = max(sigma, 1e-10 * np.mean(ub - lb))
                sigma = min(sigma, 0.5 * np.mean(ub - lb))

        return best_val, best_x