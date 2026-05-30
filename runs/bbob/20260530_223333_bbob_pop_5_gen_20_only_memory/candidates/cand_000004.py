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
        x = rng.uniform(lb, ub)
        evals = 1
        best_x = x.copy()
        best_val = func(x)
        report_best(best_val, best_x)

        # Initial step sizes (10% of range)
        step = 0.1 * (ub - lb)
        min_step = 1e-10 * (ub - lb)  # per-dim threshold

        # Placeholder for previous best before exploratory moves
        prev_x = x.copy()
        prev_val = best_val

        while evals < budget and np.any(step > min_step):
            # Exploratory moves: coordinate search from current x
            improved = False
            # Randomize order of coordinates
            order = rng.permutation(dim)
            for d in order:
                if evals >= budget:
                    break
                # Try positive direction
                x_new = x.copy()
                x_new[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val_new = func(x_new)
                    evals += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        x = x_new.copy()
                        improved = True
                        report_best(best_val, best_x)
                        continue
                # Try negative direction
                x_new = x.copy()
                x_new[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val_new = func(x_new)
                    evals += 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        x = x_new.copy()
                        improved = True
                        report_best(best_val, best_x)

            if improved:
                # Pattern move: extend in direction of improvement
                direction = x - prev_x
                x_pattern = x + direction
                x_pattern = np.clip(x_pattern, lb, ub)
                if np.any(x_pattern != x):
                    val_pattern = func(x_pattern)
                    evals += 1
                    if val_pattern < best_val:
                        best_val = val_pattern
                        best_x = x_pattern.copy()
                        x = x_pattern.copy()
                        improved = True
                        report_best(best_val, best_x)
                # Update previous point for next pattern move
                prev_x = x.copy()
                prev_val = best_val
            else:
                # No improvement: shrink step
                step *= 0.5
                # Reset x to best (optional, but helps in rugged landscapes)
                x = best_x.copy()
                prev_x = x.copy()
                prev_val = best_val

        # Exhaust remaining budget with random sampling if any left
        while evals < budget:
            x_rand = rng.uniform(lb, ub)
            val_rand = func(x_rand)
            evals += 1
            if val_rand < best_val:
                best_val = val_rand
                best_x = x_rand.copy()
                report_best(best_val, best_x)

        return best_val, best_x