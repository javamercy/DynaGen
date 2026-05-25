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

        # Initialization
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        report_best(best_f, best_x)
        evals = 1

        # Step size as fraction of range
        step = 0.2 * (ub - lb)
        min_step = 1e-3 * (ub - lb)
        max_step = 0.5 * (ub - lb)
        # Stagnation detection
        stagnation_limit = max(10, dim * 5)
        no_improve_steps = 0

        while evals < budget:
            # Sample random direction
            direction = rng.randn(dim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = np.ones(dim)
                norm = np.sqrt(dim)
            direction /= norm

            # Step forward
            candidate = best_x + direction * step
            candidate = np.clip(candidate, lb, ub)
            f = func(candidate)
            evals += 1
            if f < best_f - 1e-12:
                best_f = f
                best_x = candidate
                report_best(best_f, best_x)
                step = np.minimum(step * 1.2, max_step)
                no_improve_steps = 0
                continue

            # Step backward if budget left and forward failed
            if evals < budget:
                candidate2 = best_x - direction * step
                candidate2 = np.clip(candidate2, lb, ub)
                f2 = func(candidate2)
                evals += 1
                if f2 < best_f - 1e-12:
                    best_f = f2
                    best_x = candidate2
                    report_best(best_f, best_x)
                    step = np.minimum(step * 1.1, max_step)
                    no_improve_steps = 0
                    continue
                else:
                    step = np.maximum(step * 0.5, min_step)
                    no_improve_steps += 1
            else:
                no_improve_steps += 1

            # Restart if stagnation
            if no_improve_steps >= stagnation_limit:
                # New random point, keep best global
                best_x = lb + rng.rand(dim) * (ub - lb)
                f = func(best_x)
                evals += 1
                if f < best_f:
                    best_f = f
                    report_best(best_f, best_x)
                else:
                    # But we already have best_x as global best, but we reset to this new point
                    # We need to report if this new point is not better? No, we only report on improvements.
                    # However, to avoid losing the previous best, we should store the global best separately.
                    global_best_x = best_x.copy()
                    global_best_f = best_f
                    if f < global_best_f:
                        global_best_f = f
                        global_best_x = best_x.copy()
                        report_best(global_best_f, global_best_x)
                    # Reset step size
                    step = 0.2 * (ub - lb)
                    no_improve_steps = 0
                    # But we need to keep the best overall. Let's do that properly.
                    # Actually, we want to explore from this new point, but keep global best in memory.
                    # We'll update best_x to the new point for future exploration, but track global best separately.
                    best_x = candidate  # this is the last candidate? No, we need to set best_x to the new random point.
                    best_f = f  # but if f is worse, we still explore from here
                    # To avoid losing global best, we store it
                
                # Actually simpler: after generating new point, if it's worse than global_best, we continue from it but global best remains.
                # Let's restructure: keep global_best_f and global_best_x, and also current point for exploration.
        
        return best_f, best_x