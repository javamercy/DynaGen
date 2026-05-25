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

        # Initialize first point uniformly
        x0 = lb + (ub - lb) * rng.rand(dim)
        f0 = func(x0)
        best_f = f0
        best_x = x0.copy()
        report_best(best_f, best_x)
        evals = 1

        # Current incumbent for sampling
        curr_x = best_x.copy()
        curr_f = best_f

        # Adaptive radius: start with half of average box size
        radius = 0.5 * np.mean(ub - lb)
        max_stagnation = max(10, dim)
        stagnation = 0

        while evals < budget:
            # Sample candidate from Gaussian centered at curr_x
            candidate = curr_x + rng.randn(dim) * radius
            # Clip to bounds
            candidate = np.clip(candidate, lb, ub)
            f_candidate = func(candidate)
            evals += 1

            if f_candidate < curr_f:
                # Improvement: update current and global if better
                curr_f = f_candidate
                curr_x = candidate.copy()
                if curr_f < best_f:
                    best_f = curr_f
                    best_x = curr_x.copy()
                    report_best(best_f, best_x)
                # Exploitation: shrink radius
                radius *= 0.5
                stagnation = 0
            else:
                stagnation += 1
                # Expand radius on stagnation
                radius = min(np.max(ub - lb), radius * 1.5)

            # Restart if stagnation too long
            if stagnation >= max_stagnation:
                # Reset current to a new random point (global best retained)
                curr_x = lb + (ub - lb) * rng.rand(dim)
                curr_f = func(curr_x)
                evals += 1
                if curr_f < best_f:
                    best_f = curr_f
                    best_x = curr_x.copy()
                    report_best(best_f, best_x)
                # Reset radius and stagnation
                radius = 0.5 * np.mean(ub - lb)
                stagnation = 0

        return best_f, best_x