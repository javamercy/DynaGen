import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        best_val = float('inf')
        best_x = None

        # Single long restart
        x0 = rng.uniform(lb, ub)
        x = x0.copy()
        fx = func(x)
        evals = 1
        best_val = fx
        best_x = x.copy()
        report_best(best_val, best_x)

        # Larger initial step for faster progress
        step = 0.2 * (ub - lb)
        min_step = 1e-8 * (ub - lb)
        prev_x = x.copy()
        prev_val = fx

        # Pattern search
        while evals < budget and np.any(step > min_step):
            improved = False
            order = rng.permutation(dim)
            for d in order:
                if evals >= budget:
                    break
                # Positive direction
                x_new = x.copy()
                x_new[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val_new = func(x_new)
                    evals += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                    if val_new < fx:
                        fx = val_new
                        x = x_new.copy()
                        improved = True
                        continue
                # Negative direction
                x_new = x.copy()
                x_new[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val_new = func(x_new)
                    evals += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                    if val_new < fx:
                        fx = val_new
                        x = x_new.copy()
                        improved = True

            if improved:
                # Pattern move (extrapolation)
                direction = x - prev_x
                x_pattern = x + direction
                x_pattern = np.clip(x_pattern, lb, ub)
                if np.any(x_pattern != x) and evals < budget:
                    val_pattern = func(x_pattern)
                    evals += 1
                    if val_pattern < best_val:
                        best_val = val_pattern
                        best_x = x_pattern.copy()
                        report_best(best_val, best_x)
                    if val_pattern < fx:
                        fx = val_pattern
                        x = x_pattern.copy()
                prev_x = x.copy()
                prev_val = fx
            else:
                step *= 0.5
                x = best_x.copy() if best_x is not None else x0.copy()
                prev_x = x.copy()
                prev_val = fx

        # Local refinement around best with random perturbations
        remaining = budget - evals
        if remaining > 0:
            # Use a small sigma relative to bounds
            sigma = 0.01 * (ub - lb)
            for _ in range(remaining):
                perturb = rng.normal(0, sigma, size=dim)
                x_new = np.clip(best_x + perturb, lb, ub)
                val_new = func(x_new)
                evals += 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)

        return best_val, best_x