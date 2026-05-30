import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.stagnation_limit = max(1, int(budget / 20))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        rng = self.rng
        budget = self.budget
        evals = 0

        # Initial random sampling: 20% of budget
        n_init = max(2, int(0.2 * budget))
        best_x = None
        best_f = None
        for _ in range(min(n_init, budget)):
            x = rng.uniform(lb, ub, size=dim)
            f = func(x)
            evals += 1
            if best_f is None or f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Adaptive random search with restart
        scale = 0.1 * (ub - lb)  # initial scale per dimension
        last_improvement = evals
        while evals < budget:
            # Check stagnation
            if evals - last_improvement >= self.stagnation_limit:
                # Restart: new random point, reset scale
                x = rng.uniform(lb, ub, size=dim)
                f = func(x)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
                    last_improvement = evals
                scale = 0.1 * (ub - lb)  # reset scale
            else:
                # Sample perturbation around best
                perturbation = rng.randn(dim) * scale
                x = best_x + perturbation
                x = np.clip(x, lb, ub)
                f = func(x)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
                    last_improvement = evals
                    scale *= 1.2  # increase on success
                else:
                    scale *= 0.95  # decrease on failure
                # Ensure minimum scale
                scale = np.maximum(scale, 1e-10 * (ub - lb))
        return best_f, best_x