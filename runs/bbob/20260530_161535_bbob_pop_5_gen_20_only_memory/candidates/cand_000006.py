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

        # Initial random point
        x0 = rng.uniform(lb, ub, size=dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        report_best(best_f, best_x)
        evals = 1

        # Adaptive radius parameters (more exploitation-oriented)
        radius = 0.1 * (ub - lb)  # smaller initial radius
        shrink_factor = 0.8      # shrink faster
        expand_factor = 1.1      # expand slower
        stagnation_limit = max(100, int(budget * 0.1))  # longer stagnation before restart
        no_improve = 0

        while evals < budget:
            # Sample candidate around best with Gaussian perturbation
            candidate = best_x + rng.normal(0, radius, size=dim)
            candidate = np.clip(candidate, lb, ub)
            f_candidate = func(candidate)
            evals += 1

            if f_candidate < best_f:
                best_f = f_candidate
                best_x = candidate.copy()
                report_best(best_f, best_x)
                no_improve = 0
                # Expand radius on improvement (slow)
                radius = np.minimum(radius * expand_factor, (ub - lb) * 0.5)
            else:
                no_improve += 1
                # Shrink radius on failure (fast)
                radius = radius * shrink_factor
                # Prevent radius from becoming too small
                radius = np.maximum(radius, 1e-8 * (ub - lb))

            # Restart if stagnation
            if no_improve >= stagnation_limit:
                x_restart = rng.uniform(lb, ub, size=dim)
                f_restart = func(x_restart)
                evals += 1
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                    report_best(best_f, best_x)
                # Reset radius to small value after restart to promote local search
                radius = 0.1 * (ub - lb)
                no_improve = 0

        # Final local refinement with remaining budget (if any)
        # Use a very small radius for fine-tuning
        refinement_radius = 0.01 * (ub - lb)
        while evals < budget:
            candidate = best_x + rng.normal(0, refinement_radius, size=dim)
            candidate = np.clip(candidate, lb, ub)
            f_candidate = func(candidate)
            evals += 1
            if f_candidate < best_f:
                best_f = f_candidate
                best_x = candidate.copy()
                report_best(best_f, best_x)
                # Optionally shrink refinement radius further
                refinement_radius *= 0.9
                refinement_radius = np.maximum(refinement_radius, 1e-10 * (ub - lb))

        return best_f, best_x