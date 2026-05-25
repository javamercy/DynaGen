import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)
        scale = np.mean(ub - lb)

        # Initial random sampling to find a good starting point
        n_init = max(1, int(0.1 * budget))
        best_val = float('inf')
        best_x = None
        for _ in range(n_init):
            x = lb + (ub - lb) * rng.rand(dim)
            val = func(x)
            budget -= 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if budget <= 0:
                return best_val, best_x

        # Pattern search starting from best initial point
        x = best_x.copy()
        step = 0.1 * scale
        step_min = 1e-12 * scale
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)

        while budget > 0:
            improved = False
            for d in directions:
                if budget <= 0:
                    break
                candidate = np.clip(x + step * d, lb, ub)
                val = func(candidate)
                budget -= 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                    x = candidate
                    improved = True
                    step *= 1.2
                    break
            if not improved:
                step *= 0.5
                if step < step_min:
                    # Local restart around best point
                    x = best_x + 0.1 * scale * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    budget -= 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                    else:
                        x = best_x.copy()
                    step = 0.1 * scale
            if budget <= 0:
                break

        return best_val, best_x