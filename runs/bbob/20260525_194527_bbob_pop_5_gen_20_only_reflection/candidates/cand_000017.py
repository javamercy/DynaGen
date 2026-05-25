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
        # Initial point
        x = lb + self.rng.rand(self.dim) * (ub - lb)
        x = np.clip(x, lb, ub)
        best_x = x.copy()
        best_val = func(x)
        report_best(best_val, best_x)
        calls = 1

        # Step sizes per dimension
        step = 0.1 * (ub - lb)
        step_min = 1e-12 * (ub - lb)
        step_max = 0.5 * (ub - lb)
        shrink = 0.5
        expand = 1.5

        max_stagnation = 3 * self.dim
        stagnation = 0

        while calls < self.budget:
            improved = False
            base_x = best_x.copy()

            # Coordinate search
            for i in range(self.dim):
                if calls >= self.budget:
                    break
                # Positive step
                x_pos = best_x.copy()
                x_pos[i] += step[i]
                x_pos = np.clip(x_pos, lb, ub)
                if not np.array_equal(x_pos, best_x):
                    val_pos = func(x_pos)
                    calls += 1
                    if val_pos < best_val:
                        best_val = val_pos
                        best_x = x_pos.copy()
                        report_best(best_val, best_x)
                        improved = True
                        continue
                # Negative step (if positive didn't improve)
                if calls >= self.budget:
                    break
                x_neg = best_x.copy()
                x_neg[i] -= step[i]
                x_neg = np.clip(x_neg, lb, ub)
                if not np.array_equal(x_neg, best_x) and not np.array_equal(x_neg, x_pos):
                    val_neg = func(x_neg)
                    calls += 1
                    if val_neg < best_val:
                        best_val = val_neg
                        best_x = x_neg.copy()
                        report_best(best_val, best_x)
                        improved = True

            # Pattern move if improvement
            if improved and calls < self.budget:
                direction = best_x - base_x
                pattern_x = best_x + direction
                pattern_x = np.clip(pattern_x, lb, ub)
                if not np.array_equal(pattern_x, best_x):
                    val_pattern = func(pattern_x)
                    calls += 1
                    if val_pattern < best_val:
                        best_val = val_pattern
                        best_x = pattern_x.copy()
                        report_best(best_val, best_x)
                        # already improved, but keep improved flag

            # Diversification: random perturbation when stagnating
            if stagnation >= 2 * self.dim and stagnation < max_stagnation and calls < self.budget:
                pert = self.rng.randn(self.dim) * step
                candidate = np.clip(best_x + pert, lb, ub)
                if not np.array_equal(candidate, best_x):
                    val_cand = func(candidate)
                    calls += 1
                    if val_cand < best_val:
                        best_val = val_cand
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        improved = True

            # Update step sizes and stagnation
            if improved:
                step = np.minimum(step * expand, step_max)
                stagnation = 0
            else:
                step = np.maximum(step * shrink, step_min)
                stagnation += 1

            # Restart if stagnated
            if stagnation >= max_stagnation and calls < self.budget:
                new_x = lb + self.rng.rand(self.dim) * (ub - lb)
                new_x = np.clip(new_x, lb, ub)
                new_val = func(new_x)
                calls += 1
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
                step = 0.1 * (ub - lb)
                stagnation = 0

        return best_val, best_x