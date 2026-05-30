import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = np.inf
        best_x = None

        # Initialization: uniform random points
        n_init = max(1, min(10, budget // 3))
        for _ in range(n_init):
            x = lb + self.rng.uniform(size=dim) * (ub - lb)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        sigma = 0.1
        max_no_improve = 10 * dim
        no_improve = 0
        force_global = False

        while calls < budget:
            if force_global or (self.rng.uniform() < 0.1) or (best_x is None):
                # Global random sampling
                x = lb + self.rng.uniform(size=dim) * (ub - lb)
                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
                    no_improve = 0
                else:
                    no_improve += 1
                force_global = False
            else:
                # Local perturbation around best
                direction = self.rng.normal(0, 1, size=dim)
                step = sigma * (ub - lb) * direction
                x = best_x + step
                x = np.clip(x, lb, ub)
                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
                    sigma = min(0.5, sigma * 1.2)
                    no_improve = 0
                else:
                    sigma = max(1e-4, sigma * 0.9)
                    no_improve += 1

            if no_improve >= max_no_improve:
                force_global = True
                sigma = 0.1
                no_improve = 0

        return best_f, best_x