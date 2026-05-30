import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
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

        # Evaluate initial points: center and 3 random points
        n_initial = min(4, budget)
        points = []
        # center
        center = (lb + ub) / 2.0
        points.append(center)
        # random points
        for _ in range(n_initial - 1):
            x = rng.uniform(lb, ub)
            points.append(x)

        best_x = None
        best_val = np.inf
        fevals = 0
        for x in points:
            if fevals >= budget:
                break
            val = func(x)
            fevals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        if best_x is None:
            # fallback (shouldn't happen)
            best_x = center.copy()
            best_val = func(best_x)
            fevals += 1
            report_best(best_val, best_x)

        # Adaptive random search with restarts
        sigma = 0.2 * (ub - lb).mean()  # initial step size
        n_restart = 0
        max_stall = max(10, dim * 2)  # number of failures before restart
        stall_count = 0

        while fevals < budget:
            # Sample candidate point around best_x
            x_candidate = best_x + sigma * rng.randn(dim)
            # Clip to bounds
            x_candidate = np.clip(x_candidate, lb, ub)
            # Evaluate
            val = func(x_candidate)
            fevals += 1
            if val < best_val:
                best_val = val
                best_x = x_candidate.copy()
                sigma = sigma * 1.2  # increase step on success
                stall_count = 0
                report_best(best_val, best_x)
            else:
                stall_count += 1
                sigma = sigma * 0.95  # decrease step on failure

            # Restart if stalled
            if stall_count >= max_stall and fevals < budget:
                # Restart: sample a new random point far from current best
                x_new = rng.uniform(lb, ub)
                # With some probability, also perturb the best
                if rng.rand() < 0.5:
                    x_new = best_x + (ub - lb) * 0.5 * rng.randn(dim)
                    x_new = np.clip(x_new, lb, ub)
                val_new = func(x_new)
                fevals += 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
                    sigma = 0.2 * (ub - lb).mean()  # reset sigma
                else:
                    # Keep best unchanged, just reset sigma
                    sigma = 0.2 * (ub - lb).mean()
                stall_count = 0
                n_restart += 1

        return best_val, best_x