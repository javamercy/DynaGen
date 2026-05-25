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

        # Initial random sampling
        n_init = max(1, int(0.2 * budget))
        best_val = float('inf')
        best_x = None
        for _ in range(n_init):
            if budget <= 0:
                break
            x = lb + (ub - lb) * rng.rand(dim)
            val = func(x)
            budget -= 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        if budget <= 0:
            return best_val, best_x

        # Start pattern search from best point
        x = best_x.copy()
        step = 0.1 * scale
        step_min = 1e-12 * scale
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)

        no_improve_count = 0
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
                    no_improve_count = 0
                    break
            if not improved:
                step *= 0.5
                no_improve_count += 1
                if step < step_min or no_improve_count > 10 * dim:
                    if rng.rand() < 0.5:
                        # Local restart: perturb best
                        new_x = best_x + 0.1 * scale * rng.randn(dim)
                        new_x = np.clip(new_x, lb, ub)
                        val = func(new_x)
                        budget -= 1
                        if val < best_val:
                            best_val = val
                            best_x = new_x.copy()
                            report_best(best_val, best_x)
                            x = new_x
                        else:
                            x = best_x.copy()
                    else:
                        # Global restart: new random point
                        new_x = lb + (ub - lb) * rng.rand(dim)
                        val = func(new_x)
                        budget -= 1
                        if val < best_val:
                            best_val = val
                            best_x = new_x.copy()
                            report_best(best_val, best_x)
                        x = best_x.copy()
                    step = 0.1 * scale
                    no_improve_count = 0
            if budget <= 0:
                break
        return best_val, best_x