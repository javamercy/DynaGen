import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial point: random
        x = np.random.uniform(lb, ub)
        best_val = func(x)
        best_x = x.copy()
        evals = 1
        report_best(best_val, best_x)

        # Local search parameters
        sigma = 0.2 * (ub - lb).mean()
        success_history = []
        history_len = 20

        while evals < budget:
            # Generate trial by perturbing best
            trial = best_x + np.random.normal(0, sigma, dim)
            trial = np.clip(trial, lb, ub)
            val = func(trial)
            evals += 1
            improvement = val < best_val
            if improvement:
                best_val = val
                best_x = trial.copy()
                report_best(best_val, best_x)

            # Update success history
            success_history.append(improvement)
            if len(success_history) > history_len:
                success_history.pop(0)

            # Adapt sigma every 10 evaluations when enough history
            if evals % 10 == 0 and len(success_history) >= history_len:
                success_rate = sum(success_history) / len(success_history)
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-8, (ub - lb).mean())

            # Occasional random exploration
            if np.random.rand() < 0.01 and evals < budget:
                trial = np.random.uniform(lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                success_history.append(val < best_val)
                if len(success_history) > history_len:
                    success_history.pop(0)

        return best_val, best_x