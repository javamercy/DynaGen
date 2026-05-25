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
        # initial random point
        x = lb + self.rng.rand(self.dim) * (ub - lb)
        best_x = x.copy()
        best_val = func(x)
        report_best(best_val, best_x)
        calls = 1

        step = 0.1 * (ub - lb)
        step_min = 1e-12 * (ub - lb)
        step_max = 0.5 * (ub - lb)
        shrink = 0.5
        expand = 1.5

        stagnation = 0
        max_stagnation = 3
        restart_count = 0
        max_restarts = 3

        while calls < self.budget:
            improved = False
            base_x = best_x.copy()

            # coordinate search
            for i in range(self.dim):
                if calls >= self.budget:
                    break
                # positive step
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
                # negative step
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

            # pattern move
            if improved:
                direction = best_x - base_x
                pattern_x = best_x + direction
                pattern_x = np.clip(pattern_x, lb, ub)
                if calls < self.budget and not np.array_equal(pattern_x, best_x):
                    val_pattern = func(pattern_x)
                    calls += 1
                    if val_pattern < best_val:
                        best_val = val_pattern
                        best_x = pattern_x.copy()
                        report_best(best_val, best_x)
                step = np.minimum(step * expand, step_max)
                stagnation = 0
            else:
                step = np.maximum(step * shrink, step_min)
                stagnation += 1
                if stagnation >= max_stagnation and restart_count < max_restarts:
                    if calls < self.budget:
                        new_x = lb + self.rng.rand(self.dim) * (ub - lb)
                        new_val = func(new_x)
                        calls += 1
                        if new_val < best_val:
                            best_val = new_val
                            best_x = new_x.copy()
                            report_best(best_val, best_x)
                        stagnation = 0
                        restart_count += 1

            # early stop if steps too small
            if np.all(step < step_min * 10):
                break

        return best_val, best_x