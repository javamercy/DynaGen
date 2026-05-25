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
        # Initial point: center of bounds
        x0 = (lb + ub) / 2.0
        best_x = x0.copy()
        best_f = func(best_x)
        n_evals = 1
        report_best(best_f, best_x)
        # Initial step sizes: 0.1 * domain range in each dimension, clipped to avoid zero
        step = np.clip(0.1 * (ub - lb), 1e-6, None)
        # Main loop
        while n_evals < self.budget:
            improved = False
            # Random permutation of coordinates (controlled by seed)
            order = self.rng.permutation(dim)
            for i in order:
                if n_evals >= self.budget:
                    break
                # Positive step
                x_candidate = best_x.copy()
                x_candidate[i] = np.clip(x_candidate[i] + step[i], lb[i], ub[i])
                f_candidate = func(x_candidate)
                n_evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = x_candidate
                    improved = True
                    report_best(best_f, best_x)
                    continue  # skip negative step if positive succeeded
                # Negative step
                x_candidate = best_x.copy()
                x_candidate[i] = np.clip(x_candidate[i] - step[i], lb[i], ub[i])
                f_candidate = func(x_candidate)
                n_evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = x_candidate
                    improved = True
                    report_best(best_f, best_x)
            # Pattern move: if any coordinate step improved, try moving in the sum direction
            if improved and n_evals < self.budget:
                # Compute pattern step as the vector from initial best to new best (accumulated changes)
                # But we only have best_x; we can compute a pattern direction as the net change from previous best?
                # Simpler: try a diagonal move adding step to all coordinates
                x_candidate = np.clip(best_x + step, lb, ub)
                f_candidate = func(x_candidate)
                n_evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = x_candidate
                    improved = True
                    report_best(best_f, best_x)
            # If no improvement in this cycle, reduce step sizes
            if not improved:
                step *= 0.5
                # Ensure step not too small
                step = np.maximum(step, 1e-12)
        return best_f, best_x