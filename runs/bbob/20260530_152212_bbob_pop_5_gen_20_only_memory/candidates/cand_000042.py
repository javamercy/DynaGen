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
        # initial random sampling: 10% of budget
        n_init = max(2, int(0.1 * self.budget))
        for _ in range(min(n_init, self.budget)):
            x = np.random.uniform(lb, ub, size=self.dim)
            f = func(x)
            self.evals += 1
            if best_f is None or f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        # first pattern search with moderate step
        step_init = 0.2 * (ub - lb).mean()
        step = step_init
        while self.evals < self.budget and step > 1e-12:
            improved = False
            for i in range(self.dim):
                if self.evals >= self.budget:
                    break
                # positive direction
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
                # negative direction
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
                step *= 0.5
        # second pattern search with fine step if budget remains
        if self.evals < self.budget:
            step_fine = 1e-4 * (ub - lb).mean()
            step = step_fine
            while self.evals < self.budget and step > 1e-15:
                improved = False
                for i in range(self.dim):
                    if self.evals >= self.budget:
                        break
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
                    step *= 0.5
        # final random perturbations if any budget left
        while self.evals < self.budget:
            x_try = best_x + np.random.normal(0, step_fine * 0.1, size=self.dim)
            x_try = np.clip(x_try, lb, ub)
            f_try = func(x_try)
            self.evals += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
                report_best(best_f, best_x)
        return best_f, best_x