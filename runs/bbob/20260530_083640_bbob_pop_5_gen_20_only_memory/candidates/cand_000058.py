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
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        current_x = best_x.copy()
        current_val = best_val

        # Phase 1: exploration (first 70% of budget)
        phase1_budget = int(0.7 * budget)
        T0 = 1.0
        T_end_explore = 1e-2
        step0 = 0.2 * (ub - lb)
        step_end_explore = 1e-3 * (ub - lb)

        while calls < phase1_budget:
            t = (calls - 1) / (phase1_budget - 1) if phase1_budget > 1 else 1.0
            T = T0 * (T_end_explore / T0) ** t
            step = step0 * (step_end_explore / step0) ** t

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

        # Phase 2: exploitation (remaining budget)
        # Use small step around best point
        step0_exploit = 1e-2 * (ub - lb)
        step_end_exploit = 1e-5 * (ub - lb)
        current_x = best_x.copy()
        current_val = best_val

        while calls < budget:
            t = (calls - phase1_budget) / (budget - phase1_budget - 1) if budget - phase1_budget > 1 else 1.0
            step = step0_exploit * (step_end_exploit / step0_exploit) ** t
            T = 1e-3  # very low temperature, almost greedy

            candidate = current_x + step * rng.randn(dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            calls += 1

            if val < current_val:
                current_x = candidate
                current_val = val
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
            else:
                # accept rarely to escape local minima
                if rng.rand() < np.exp(-(val - current_val) / T):
                    current_x = candidate
                    current_val = val

        return best_val, best_x