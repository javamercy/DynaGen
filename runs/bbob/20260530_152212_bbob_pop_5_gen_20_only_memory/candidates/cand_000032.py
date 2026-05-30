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
        n_init = max(2, int(0.3 * self.budget))
        # Initial random sampling
        for _ in range(min(n_init, self.budget)):
            x = np.random.uniform(lb, ub, size=self.dim)
            f = func(x)
            self.evals += 1
            if best_f is None or f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        # Adaptive pattern search with restarts
        step_init = 0.2 * (ub - lb).mean()
        step = step_init
        no_improve_count = 0
        restart_threshold = max(10, int(0.05 * self.budget))
        while self.evals < self.budget:
            improved = False
            order = np.random.permutation(self.dim)
            for i in order:
                if self.evals >= self.budget:
                    break
                # Try positive direction
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
                # Try negative direction
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
                no_improve_count = 0
            else:
                step *= 0.5
                no_improve_count += self.dim  # each dimension tried without improvement
            if no_improve_count > restart_threshold and self.evals < self.budget - 5:
                # Restart from new random point
                x_restart = np.random.uniform(lb, ub, size=self.dim)
                f_restart = func(x_restart)
                self.evals += 1
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                    report_best(best_f, best_x)
                # Reset step
                step = step_init
                no_improve_count = 0
        # Final random perturbations
        while self.evals < self.budget:
            x_try = best_x + np.random.normal(0, step_init * 0.01, size=self.dim)
            x_try = np.clip(x_try, lb, ub)
            f_try = func(x_try)
            self.evals += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
                report_best(best_f, best_x)
        return best_f, best_x