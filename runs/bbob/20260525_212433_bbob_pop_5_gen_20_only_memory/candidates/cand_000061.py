import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        if self.budget == 0:
            return float('inf'), None
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim
        rng = self.rng

        # Phase 1: Adaptive Simulated Annealing
        phase1_budget = budget // 2
        best_val = float('inf')
        best_x = None

        # Initial point
        x = lb + (ub - lb) * rng.rand(dim)
        val = func(x)
        best_val = val
        best_x = x.copy()
        report_best(best_val, best_x)
        evals = 1

        T0 = 1.0
        T = T0
        step_size = 0.2 * (ub - lb)
        n_adapt = max(1, phase1_budget // 20)
        success_count = 0

        while evals < phase1_budget and evals < budget:
            delta = rng.randn(dim) * step_size
            y = x + delta
            y = np.clip(y, lb, ub)
            new_val = func(y)
            evals += 1
            if new_val < val:
                accept = True
            else:
                if rng.rand() < np.exp((val - new_val) / T):
                    accept = True
                else:
                    accept = False
            if accept:
                x = y
                val = new_val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                success_count += 1
            if evals % n_adapt == 0:
                success_rate = success_count / n_adapt
                if success_rate > 0.2:
                    step_size *= 1.2
                else:
                    step_size *= 0.85
                success_count = 0
            T = T0 * (1 - evals / phase1_budget)

        # Phase 2: Coordinate-wise line search around best_x
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            x_best = best_x.copy()
            val_best = best_val
            step_size_local = 0.1 * (ub - lb)  # initial step for local search
            for it in range(remaining):
                dim_idx = it % dim
                # try positive step
                step = step_size_local[dim_idx]
                y = x_best.copy()
                y[dim_idx] += step
                y[dim_idx] = np.clip(y[dim_idx], lb[dim_idx], ub[dim_idx])
                new_val = func(y)
                evals += 1
                if new_val < val_best:
                    val_best = new_val
                    x_best = y
                    best_val = val_best
                    best_x = x_best.copy()
                    report_best(best_val, best_x)
                    # increase step on success
                    step_size_local[dim_idx] *= 1.5
                else:
                    # try negative step
                    y2 = x_best.copy()
                    y2[dim_idx] -= step
                    y2[dim_idx] = np.clip(y2[dim_idx], lb[dim_idx], ub[dim_idx])
                    new_val2 = func(y2)
                    evals += 1
                    if new_val2 < val_best:
                        val_best = new_val2
                        x_best = y2
                        best_val = val_best
                        best_x = x_best.copy()
                        report_best(best_val, best_x)
                        step_size_local[dim_idx] *= 1.5
                    else:
                        # no improvement, shrink step
                        step_size_local[dim_idx] *= 0.5
                if evals >= budget:
                    break
        return best_val, best_x