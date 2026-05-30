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

        # Initial point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        parent_x = best_x.copy()
        parent_val = best_val

        # Step size: scaled by average range
        avg_range = np.mean(ub - lb)
        sigma = 0.2 * avg_range
        sigma_min = 1e-8 * avg_range
        sigma_max = 0.5 * avg_range

        # Success rate tracking
        success_count = 0
        total_in_window = 0
        window_size = min(10, budget // 10)
        if window_size < 1:
            window_size = 1
        last_improvement_evals = evals

        while evals < budget:
            # Generate offspring
            offspring = parent_x + sigma * rng.randn(dim)
            offspring = np.clip(offspring, lb, ub)
            o_val = func(offspring)
            evals += 1

            # Update best
            if o_val < best_val:
                best_val = o_val
                best_x = offspring.copy()
                report_best(best_val, best_x)

            # Update parent and success
            if o_val < parent_val:
                parent_x = offspring.copy()
                parent_val = o_val
                success_count += 1
                last_improvement_evals = evals
            total_in_window += 1

            # Adapt sigma after window
            if total_in_window >= window_size:
                success_rate = success_count / total_in_window
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                sigma = np.clip(sigma, sigma_min, sigma_max)
                success_count = 0
                total_in_window = 0

            # Restart if no improvement for a long time
            if evals - last_improvement_evals > max(dim * 10, 30):
                # Restart with random point
                parent_x = rng.uniform(lb, ub)
                parent_val = func(parent_x)
                evals += 1
                if parent_val < best_val:
                    best_val = parent_val
                    best_x = parent_x.copy()
                    report_best(best_val, best_x)
                # Reset sigma
                sigma = 0.2 * avg_range
                success_count = 0
                total_in_window = 0
                last_improvement_evals = evals

            if evals >= budget:
                break

        return best_val, best_x