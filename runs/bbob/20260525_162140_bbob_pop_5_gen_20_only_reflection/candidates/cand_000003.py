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

        # Initialization
        x0 = lb + rng.rand(dim) * (ub - lb)
        f0 = func(x0)
        best_x = x0
        best_f = f0
        report_best(best_f, best_x)
        evals = 1

        # Parameters
        init_radius = 0.2 * np.mean(ub - lb)
        radius = init_radius
        stagnation_limit = max(10, dim * 10)
        no_improve_steps = 0
        success_window = 5
        success_count = 0

        while evals < budget:
            # Sample candidate
            candidate = best_x + rng.uniform(-radius, radius, dim)
            candidate = np.clip(candidate, lb, ub)
            f = func(candidate)
            evals += 1

            if f < best_f - 1e-12:
                best_f = f
                best_x = candidate
                report_best(best_f, best_x)
                no_improve_steps = 0
                success_count += 1
            else:
                no_improve_steps += 1

            # Adapt radius
            if evals % success_window == 0:
                if success_count > 0.5 * success_window:
                    radius = min(radius * 1.2, 0.5 * np.mean(ub - lb))
                else:
                    radius = max(radius * 0.8, 0.01 * np.mean(ub - lb))
                success_count = 0

            # Restart if stagnation
            if no_improve_steps >= stagnation_limit:
                # Reset radius and recenter at best
                radius = init_radius
                no_improve_steps = 0
                # optionally sample a random point to escape? but keep best
                # We'll just continue from best with new radius

        return best_f, best_x