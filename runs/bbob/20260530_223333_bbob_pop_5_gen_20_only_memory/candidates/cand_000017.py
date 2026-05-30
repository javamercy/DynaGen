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

        # Initial point
        x = rng.uniform(lb, ub)
        fx = func(x)
        evals = 1
        if fx < best_val:
            best_val = fx
            best_x = x.copy()
            report_best(best_val, best_x)

        step = 0.3 * (ub - lb)
        min_step = 1e-12 * (ub - lb)
        prev_x = x.copy()
        prev_val = fx

        while evals < budget:
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
                        step[d] *= 2.0
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
                        step[d] *= 2.0
                    else:
                        step[d] *= 0.5

            if evals >= budget:
                break

            if improved:
                # Pattern step along net direction
                direction = x - prev_x
                x_pattern = x + direction
                x_pattern = np.clip(x_pattern, lb, ub)
                if np.any(x_pattern != x):
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
                # Shrink all steps and reset to best if failed
                step = np.clip(step * 0.5, min_step, None)
                if np.all(step <= min_step) or (best_x is not None and rng.rand() < 0.1):
                    # Switch to local random search around best
                    while evals < budget:
                        pert = rng.randn(dim) * step.mean() * 10.0
                        x_new = np.clip(best_x + pert, lb, ub)
                        val_new = func(x_new)
                        evals += 1
                        if val_new < best_val:
                            best_val = val_new
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                        # Update step based on success
                        if val_new < fx:
                            step = step * 2.0
                            x = x_new.copy()
                            fx = val_new
                            break
                    else:
                        step = step * 0.5
                    continue
                x = best_x.copy() if best_x is not None else x.copy()
                prev_x = x.copy()
                prev_val = fx if fx is not None else np.inf

            # Ensure we don't get stuck
            if evals >= budget:
                break

        return best_val, best_x