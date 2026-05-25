import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_y = np.inf
        self.evals = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Phase 1: Global sampling
        n_init = max(10, min(budget // 2, 5 * dim))
        for _ in range(n_init):
            if self.evals >= budget:
                return self.best_y, self.best_x
            x = lb + (ub - lb) * rng.rand(dim)
            y = func(x)
            self.evals += 1
            if y < self.best_y:
                self.best_y = y
                self.best_x = x.copy()
                report_best(y, x)

        # Phase 2: Local refinement with occasional global jumps
        step = (ub - lb) / 4.0
        step_min = 1e-6 * (ub - lb).max()
        patience = 10
        fail_count = 0

        while self.evals < budget:
            # Global jump with 10% probability
            if rng.rand() < 0.1:
                if self.evals >= budget:
                    break
                x = lb + (ub - lb) * rng.rand(dim)
                y = func(x)
                self.evals += 1
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = x.copy()
                    report_best(y, x)
                    step = (ub - lb) / 4.0
                continue

            # Local perturbation
            direction = rng.randn(dim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = np.ones(dim)
                norm = np.sqrt(dim)
            direction = direction / norm

            candidate = self.best_x + step * direction
            candidate = np.clip(candidate, lb, ub)
            if self.evals >= budget:
                break
            y = func(candidate)
            self.evals += 1
            if y < self.best_y:
                self.best_y = y
                self.best_x = candidate.copy()
                report_best(y, candidate)
                step *= 1.2
                fail_count = 0
                continue

            # Try opposite direction
            candidate = self.best_x - step * direction
            candidate = np.clip(candidate, lb, ub)
            if self.evals >= budget:
                break
            y = func(candidate)
            self.evals += 1
            if y < self.best_y:
                self.best_y = y
                self.best_x = candidate.copy()
                report_best(y, candidate)
                step *= 1.2
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= patience:
                    step /= 2.0
                    fail_count = 0
                    if step < step_min:
                        step = (ub - lb) / 4.0

        return self.best_y, self.best_x