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

        # Global exploration phase (20% of budget, at least 1)
        n_global = max(1, int(0.2 * budget))
        best_x = None
        best_val = float('inf')
        evals = 0
        for _ in range(n_global):
            if evals >= budget:
                break
            x = lb + (ub - lb) * rng.rand(dim)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # If no evaluations (budget 0), return dummy
        if evals == 0:
            x = lb + (ub - lb) * rng.rand(dim)
            best_val = func(x)
            best_x = x.copy()
            report_best(best_val, best_x)
            return (best_val, best_x)

        # Initialize step sizes and success rates
        step = 0.2 * (ub - lb)  # per-dimension step sizes
        success_rate = 0.5 * np.ones(dim)  # smoothed success rate per dimension
        alpha = 0.2  # smoothing factor
        stagnation = 0
        max_stagnation = max(5, int(0.1 * budget))
        min_step_factor = 1e-6  # relative to range

        while evals < budget:
            # Generate candidate
            perturbation = rng.normal(0, 1, dim) * step
            x_cand = best_x + perturbation
            x_cand = np.clip(x_cand, lb, ub)
            val_cand = func(x_cand)
            evals += 1
            if evals > budget:
                break

            success = np.zeros(dim, dtype=bool)
            if val_cand < best_val:
                best_val = val_cand
                best_x = x_cand.copy()
                report_best(best_val, best_x)
                success[:] = True
                stagnation = 0
            else:
                stagnation += 1

            # Update success rates per dimension (using direction of perturbation)
            for d in range(dim):
                if success[d]:
                    success_rate[d] = (1 - alpha) * success_rate[d] + alpha * 1.0
                else:
                    success_rate[d] = (1 - alpha) * success_rate[d] + alpha * 0.0

            # Adjust step sizes
            for d in range(dim):
                if success_rate[d] > 0.44:
                    step[d] *= 1.2
                else:
                    step[d] *= 0.9
                # Clamp step sizes
                step[d] = np.clip(step[d], 1e-10, 0.5 * (ub[d] - lb[d]))

            # Restart condition: if max step size too small or stagnation
            max_step = np.max(step)
            range_scale = np.max(ub - lb)
            if max_step < min_step_factor * range_scale or stagnation >= max_stagnation:
                if evals < budget:
                    x_new = lb + (ub - lb) * rng.rand(dim)
                    val_new = func(x_new)
                    evals += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                    # Reset step sizes and success rates
                    step = 0.2 * (ub - lb)
                    success_rate[:] = 0.5
                    stagnation = 0
                else:
                    break

        return (best_val, best_x)