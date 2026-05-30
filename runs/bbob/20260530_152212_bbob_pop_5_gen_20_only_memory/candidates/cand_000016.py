import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.evals = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_x = None
        best_f = None
        # initial random sampling: 5% of budget
        n_init = max(2, int(0.05 * self.budget))
        for _ in range(min(n_init, self.budget)):
            x = np.random.uniform(lb, ub, size=self.dim)
            f = func(x)
            self.evals += 1
            if best_f is None or f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        # pattern search (Hooke-Jeeves style) with coordinate sweeps
        step_init = 0.05 * (ub - lb).mean()
        step = step_init
        shrink = 0.5
        while self.evals < self.budget and step > 1e-14:
            improved = False
            # full sweep over coordinates
            for i in range(self.dim):
                if self.evals >= self.budget:
                    break
                # try positive direction
                x_try = best_x.copy()
                x_try[i] += step
                x_try[i] = np.clip(x_try[i], lb[i], ub[i])
                f_try = func(x_try)
                self.evals += 1
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    report_best(best_f, best_x)
                    improved = True
                    break
                # try negative direction
                x_try2 = best_x.copy()
                x_try2[i] -= step
                x_try2[i] = np.clip(x_try2[i], lb[i], ub[i])
                f_try2 = func(x_try2)
                self.evals += 1
                if f_try2 < best_f:
                    best_f = f_try2
                    best_x = x_try2.copy()
                    report_best(best_f, best_x)
                    improved = True
                    break
            if improved:
                step *= 2.0
            else:
                step *= shrink
        # remaining budget: small random perturbations with decaying radius
        radius = step_init * 0.01
        while self.evals < self.budget:
            scale = radius * (1.0 - (self.evals / self.budget))
            if scale < 1e-14:
                break
            x_try = best_x + np.random.normal(0, scale, size=self.dim)
            x_try = np.clip(x_try, lb, ub)
            f_try = func(x_try)
            self.evals += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
                report_best(best_f, best_x)
        return best_f, best_x