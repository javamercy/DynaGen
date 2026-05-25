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
        rang = ub - lb

        # Initial point: random within bounds
        x = lb + self.rng.rand(self.dim) * rang
        best_x = x.copy()
        best_val = func(best_x)
        report_best(best_val, best_x)
        calls = 1

        # Step sizes per dimension: initial 10% of range
        step = 0.1 * rang
        step_min = 1e-12 * rang
        step_max = 0.5 * rang
        shrink = 0.5
        expand = 1.5

        stagnation_limit = max(10, self.budget // 20)
        stagnation_counter = 0

        while calls < self.budget:
            improved = False
            base_x = best_x.copy()

            # Coordinate search (exploratory moves)
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
                        continue  # skip negative if positive improved
                # Negative step (if positive did not improve or was skipped)
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

            # Pattern move if improvement found
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
                        improved = True

            # Adapt step sizes
            if improved:
                step = np.minimum(step * expand, step_max)
                stagnation_counter = 0
            else:
                step = np.maximum(step * shrink, step_min)
                stagnation_counter += 1

            # Restart if stagnation
            if stagnation_counter >= stagnation_limit and calls < self.budget:
                # Restart from a new random point
                new_x = lb + self.rng.rand(self.dim) * rang
                new_val = func(new_x)
                calls += 1
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
                # Reset step sizes
                step = 0.1 * rang
                stagnation_counter = 0

            # Early termination if step sizes become too small
            if np.all(step < step_min * 10):
                break

        return best_val, best_x