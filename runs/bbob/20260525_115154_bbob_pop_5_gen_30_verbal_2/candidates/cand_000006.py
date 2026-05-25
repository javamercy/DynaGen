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
        dim = self.dim
        rng = self.rng
        budget = self.budget

        # Initial LHS
        n_initial = min(budget, max(dim * 2, budget // 2))
        if n_initial < 1:
            n_initial = 1
        points = self._lhs(lb, ub, n_initial)
        for x in points:
            if self.n_calls >= budget:
                break
            y = func(x)
            self.n_calls += 1
            if y < self.best_y:
                self.best_y = y
                self.best_x = x.copy()
                report_best(y, x)

        # Adaptive random search parameters
        if self.best_x is None:
            # fallback if no initial points? never happens
            self.best_x = rng.uniform(lb, ub)
            self.best_y = func(self.best_x)
            self.n_calls += 1
            report_best(self.best_y, self.best_x)

        scale = 0.2 * (ub - lb)
        stagnation_limit = max(1, budget // 10)
        no_improve = 0

        while self.n_calls < budget:
            # Perturb best
            dx = rng.normal(0, scale, size=dim)
            candidate = self.best_x + dx
            candidate = np.clip(candidate, lb, ub)
            y = func(candidate)
            self.n_calls += 1

            if y < self.best_y:
                self.best_y = y
                self.best_x = candidate.copy()
                report_best(y, candidate)
                no_improve = 0
                scale = np.maximum(scale * 0.5, 1e-8 * (ub - lb))
            else:
                no_improve += 1
                if no_improve >= stagnation_limit:
                    if self.n_calls >= budget:
                        break
                    # Restart random point
                    new_x = rng.uniform(lb, ub)
                    new_y = func(new_x)
                    self.n_calls += 1
                    if new_y < self.best_y:
                        self.best_y = new_y
                        self.best_x = new_x.copy()
                        report_best(new_y, new_x)
                    no_improve = 0
                    scale = 0.2 * (ub - lb)
                else:
                    # Expand scale periodically
                    if no_improve % max(1, stagnation_limit // 4) == 0:
                        scale = np.minimum(scale * 1.5, ub - lb)

        return self.best_y, self.best_x

    def _lhs(self, lb, ub, n):
        points = np.zeros((n, self.dim))
        for i in range(self.dim):
            strata = np.linspace(lb[i], ub[i], n+1)[:-1]
            order = self.rng.permutation(n)
            points[:, i] = strata[order] + self.rng.uniform(0, (ub[i]-lb[i])/n, size=n)
        return points