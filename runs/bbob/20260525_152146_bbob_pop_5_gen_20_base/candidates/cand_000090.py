import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        # Initialization
        best_x = rng.uniform(lb, ub, size=dim)
        best_f = func(best_x)
        budget -= 1
        report_best(best_f, best_x)

        if budget <= 0:
            return best_f, best_x

        # Current point
        x_cur = best_x.copy()
        f_cur = best_f

        # Parameters
        T0 = 1.0
        cooling = 0.99
        step_init = np.mean(ub - lb) * 0.2  # initial step size
        T = T0
        step = step_init

        stagnation_limit = max(1, budget // 10)
        no_improve_count = 0

        while budget > 0:
            # Generate candidate by perturbing each coordinate
            perturbation = rng.randn(dim) * step
            x_cand = x_cur + perturbation
            # Clip to bounds
            x_cand = np.clip(x_cand, lb, ub)
            f_cand = func(x_cand)
            budget -= 1

            delta = f_cand - f_cur

            # Acceptance criterion
            if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-10)):
                x_cur = x_cand
                f_cur = f_cand

            # Update best
            if f_cand < best_f:
                best_f = f_cand
                best_x = x_cand.copy()
                report_best(best_f, best_x)
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Cooling
            T *= cooling
            step = step_init * (T / T0)  # scale step with temperature

            # Check for restart
            if no_improve_count >= stagnation_limit and budget > 0:
                # Restart with a new random point, preserve best
                x_cur = rng.uniform(lb, ub, size=dim)
                f_cur = func(x_cur)
                budget -= 1
                # Reset temperature and step
                T = T0
                step = step_init
                no_improve_count = 0
                # Update best if the new point is better
                if f_cur < best_f:
                    best_f = f_cur
                    best_x = x_cur.copy()
                    report_best(best_f, best_x)

        return best_f, best_x