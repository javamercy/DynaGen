import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # initial point
        x = rng.uniform(lb, ub, size=dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        report_best(best_f, best_x)
        calls = 1

        range_mean = np.mean(ub - lb)
        sigma = 0.3 * range_mean
        success_counter = 0
        eval_window = 3
        local_restart_threshold = 1e-9 * range_mean

        # budget allocation: first 30% for global search, then local refinement
        global_budget = int(0.3 * budget)
        local_budget = budget - global_budget

        # Global phase
        while calls < global_budget and calls < budget:
            y = x + sigma * rng.randn(dim)
            y = np.clip(y, lb, ub)
            if calls >= budget:
                break
            fy = func(y)
            calls += 1
            if fy < fx:
                x = y.copy()
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    report_best(best_f, best_x)
                success_counter += 1

            if (calls % eval_window) == 0:
                success_rate = success_counter / eval_window
                if success_rate > 0.2:
                    sigma *= 1.3
                elif success_rate < 0.2:
                    sigma *= 0.7
                success_counter = 0

            if sigma < local_restart_threshold or (calls % (global_budget // 5 + 1) == 0 and calls > 0):
                x = best_x + 0.6 * range_mean * rng.randn(dim)
                x = np.clip(x, lb, ub)
                if calls < budget:
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        report_best(best_f, best_x)
                sigma = 0.2 * range_mean
                success_counter = 0

        # Local refinement phase
        sigma_local = 0.1 * range_mean
        while calls < budget:
            # sample near best point
            y = best_x + sigma_local * rng.randn(dim)
            y = np.clip(y, lb, ub)
            if calls >= budget:
                break
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                report_best(best_f, best_x)
                # on improvement, keep sigma same; else decrease
            else:
                sigma_local *= 0.9
            # if sigma gets too small, restart from best with moderate sigma
            if sigma_local < 1e-10 * range_mean:
                sigma_local = 0.1 * range_mean

        return best_f, best_x