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
        # Initial global sampling (30% of budget)
        n_init = max(10, int(self.budget * 0.3))
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
        # (1+1)-ES with line search and restarts
        sigma = np.mean(ub - lb) * 0.1
        success_counter = 0
        failure_counter = 0
        update_freq = 10
        iterations = 0
        stagnation_threshold = 30
        min_restart_budget = 10
        while calls < self.budget:
            # Restart if stagnation and enough budget left
            if failure_counter >= stagnation_threshold and (self.budget - calls) >= min_restart_budget:
                x = self.rng.uniform(lb, ub, size=self.dim)
                val = func(x)
                calls += 1
                if val < best_value:
                    best_value = val
                    best_x = x.copy()
                    report_best(best_value, best_x)
                sigma = np.mean(ub - lb) * 0.1
                success_counter = 0
                failure_counter = 0
                iterations = 0
                continue
            # Generate candidate
            candidate = best_x + self.rng.normal(0, sigma, size=self.dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            calls += 1
            iterations += 1
            if val < best_value:
                # Line search along direction
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
                best_value = val
                best_x = candidate.copy()
                report_best(best_value, best_x)
                success_counter += 1
                failure_counter = 0
            else:
                failure_counter += 1
            # Step-size adaptation
            if iterations % update_freq == 0:
                success_rate = success_counter / update_freq
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.85
                success_counter = 0
        return (best_value, best_x)