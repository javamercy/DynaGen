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

        # Phase 1: Simulated Annealing
        budget_sa = max(1, int(0.8 * budget))
        # Initial random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        current_x = best_x.copy()
        current_val = best_val

        max_iter_sa = budget_sa - 1
        if max_iter_sa > 0:
            T0 = 1.0
            T_end = 1e-4
            step0_sa = 0.1 * (ub - lb)
            step_end_sa = 1e-6 * (ub - lb)

            for i in range(max_iter_sa):
                t = i / max_iter_sa
                T = T0 * (T_end / T0) ** t
                step = step0_sa * (step_end_sa / step0_sa) ** t

                candidate = current_x + step * rng.randn(dim)
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                calls += 1

                delta = val - current_val
                if delta < 0:
                    current_x = candidate
                    current_val = val
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                else:
                    if rng.rand() < np.exp(-delta / T):
                        current_x = candidate
                        current_val = val

                if calls >= budget:
                    return best_val, best_x

        # Phase 2: Local Search (greedy, random perturbations)
        remaining = budget - calls
        if remaining > 0:
            step0_local = 0.05 * (ub - lb)
            step_end_local = 1e-6 * (ub - lb)
            for i in range(remaining):
                t = i / remaining
                step = step0_local * (step_end_local / step0_local) ** t
                candidate = best_x + step * rng.randn(dim)
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                if calls >= budget:
                    break

        return best_val, best_x