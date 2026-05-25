import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_y = np.inf
        self.n_calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        while self.n_calls < self.budget:
            n_initial = max(1, min(self.budget - self.n_calls, max(self.dim * 2, (self.budget - self.n_calls) // 2)))
            if n_initial < 1:
                break
            points = self._lhs(lb, ub, n_initial)
            for x in points:
                if self.n_calls >= self.budget:
                    break
                y = func(x)
                self.n_calls += 1
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = x.copy()
                    report_best(y, x)
            step_size = 0.2 * (ub - lb).mean()
            success_count = 0
            eval_count = 0
            last_improvement = 0
            window = max(1, int(self.dim * 5))
            while self.n_calls < self.budget:
                dx = self.rng.normal(0, step_size, size=self.dim)
                x_new = self.best_x + dx
                x_new = np.clip(x_new, lb, ub)
                y = func(x_new)
                self.n_calls += 1
                eval_count += 1
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = x_new.copy()
                    report_best(y, x_new)
                    success_count += 1
                    last_improvement = eval_count
                if eval_count % window == 0:
                    success_rate = success_count / window
                    if success_rate > 0.2:
                        step_size *= 1.5
                    else:
                        step_size *= 0.85
                    success_count = 0
                min_step = 1e-12 * (ub - lb).mean()
                if step_size < min_step or (eval_count - last_improvement) > 10 * self.dim:
                    break
        return self.best_y, self.best_x

    def _lhs(self, lb, ub, n):
        points = np.zeros((n, self.dim))
        for i in range(self.dim):
            strata = np.linspace(lb[i], ub[i], n + 1)[:-1]
            order = self.rng.permutation(n)
            points[:, i] = strata[order] + self.rng.uniform(0, (ub[i] - lb[i]) / n, size=n)
        return points