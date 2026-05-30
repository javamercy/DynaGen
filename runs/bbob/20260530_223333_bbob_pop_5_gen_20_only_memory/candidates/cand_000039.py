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
        best_value = np.inf
        best_x = None
        calls = 0

        # Initial global sampling (20% of budget)
        n_init = max(2, int(self.budget * 0.2))
        for _ in range(n_init):
            if calls >= self.budget:
                break
            x = self.rng.uniform(lb, ub, size=self.dim)
            val = func(x)
            calls += 1
            if val < best_value:
                best_value = val
                best_x = x.copy()
                report_best(best_value, best_x)

        # (1+1)-ES with step-size adaptation and restarts
        sigma = np.mean(ub - lb) * 0.1
        success_counter = 0
        update_freq = 10
        stagnation_limit = 20 * self.dim
        stagnation_counter = 0
        iterations = 0

        while calls < self.budget:
            # Generate candidate
            candidate = best_x + self.rng.normal(0, sigma, size=self.dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            calls += 1
            iterations += 1
            stagnation_counter += 1

            if val < best_value:
                # New best found: line search along direction
                direction = candidate - best_x
                for step in [0.5, 1.0, 1.5, 2.0]:
                    if calls >= self.budget:
                        break
                    x_new = best_x + step * direction
                    x_new = np.clip(x_new, lb, ub)
                    val_new = func(x_new)
                    calls += 1
                    if val_new < val:
                        val = val_new
                        candidate = x_new.copy()
                # Update best
                best_value = val
                best_x = candidate.copy()
                report_best(best_value, best_x)
                success_counter += 1
                stagnation_counter = 0

            # Step-size adaptation
            if iterations % update_freq == 0:
                success_rate = success_counter / update_freq
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.85
                success_counter = 0

            # Restart if stagnation
            if stagnation_counter >= stagnation_limit:
                # Keep best ever, but reinitialize from random point
                x_new = self.rng.uniform(lb, ub, size=self.dim)
                val_new = func(x_new)
                calls += 1
                if val_new < best_value:
                    best_value = val_new
                    best_x = x_new.copy()
                    report_best(best_value, best_x)
                # Reset parameters
                sigma = np.mean(ub - lb) * 0.1
                success_counter = 0
                stagnation_counter = 0
                iterations = 0

        return (best_value, best_x)