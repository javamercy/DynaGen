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

        # Initialization
        center = lb + rng.rand(dim) * (ub - lb)
        f_center = func(center)
        best_f = f_center
        best_x = center.copy()
        report_best(best_f, best_x)
        evals = 1

        # Parameters
        range_mean = np.mean(ub - lb)
        sigma = 0.2 * range_mean
        stagnation_limit = max(10, dim * 10)
        no_improve_steps = 0
        window_size = 5
        success_count = 0

        while evals < budget:
            # Uniform perturbation in hyper-rectangle
            candidate = center + rng.uniform(-sigma, sigma, dim)
            candidate = np.clip(candidate, lb, ub)
            f_candidate = func(candidate)
            evals += 1

            if f_candidate < best_f - 1e-12:
                best_f = f_candidate
                best_x = candidate.copy()
                center = candidate.copy()
                report_best(best_f, best_x)
                no_improve_steps = 0
                success_count += 1
            else:
                no_improve_steps += 1

            # Adapt sigma every window_size evaluations
            if evals % window_size == 0:
                success_rate = success_count / window_size
                if success_rate > 0.5:
                    sigma = min(sigma * 1.2, 0.5 * range_mean)
                else:
                    sigma = max(sigma * 0.8, 0.01 * range_mean)
                success_count = 0

            # Restart if stagnation
            if no_improve_steps >= stagnation_limit:
                # New random point as center
                center = lb + rng.rand(dim) * (ub - lb)
                f_center = func(center)
                evals += 1
                if f_center < best_f - 1e-12:
                    best_f = f_center
                    best_x = center.copy()
                    report_best(best_f, best_x)
                no_improve_steps = 0
                sigma = 0.2 * range_mean

        return best_f, best_x