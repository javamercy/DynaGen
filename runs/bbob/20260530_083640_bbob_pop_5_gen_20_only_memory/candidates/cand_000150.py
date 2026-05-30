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

        # Initial point
        x = lb + rng.rand(dim) * (ub - lb)
        f = func(x)
        best_x = x.copy()
        best_f = f
        evals = 1
        report_best(best_f, best_x)

        step_scale = ub - lb
        step = 0.2 * step_scale
        init_temp = 100.0
        final_temp = 1e-5
        temp = init_temp
        max_stag = max(10, budget // 10)
        stag_count = 0
        total_iters = budget - 1

        for it in range(total_iters):
            if evals >= budget:
                break
            frac = (it + 1) / total_iters
            temp = init_temp * (final_temp / init_temp) ** frac

            x_new = x + step * rng.randn(dim)
            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new)
            evals += 1

            if f_new < f:
                x = x_new.copy()
                f = f_new
                step = np.clip(step * 1.1, 1e-10 * step_scale, step_scale)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
                stag_count = 0
            else:
                delta = f_new - f
                prob = 1.0 if delta <= 0 else (np.exp(-delta / temp) if temp > 0 else 0.0)
                if rng.rand() < prob:
                    x = x_new.copy()
                    f = f_new
                    step = np.clip(step * 1.1, 1e-10 * step_scale, step_scale)
                else:
                    step = np.clip(step * 0.9, 1e-10 * step_scale, step_scale)
                stag_count += 1

            if stag_count >= max_stag and evals < budget:
                temp = init_temp
                stag_count = 0
                x = best_x.copy()
                f = best_f
                step = 0.2 * step_scale

        return best_f, best_x