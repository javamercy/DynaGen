import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # initial random point
        x0 = lb + self.rng.rand(dim) * (ub - lb)
        f0 = func(x0)
        calls += 1
        best_x = x0.copy()
        best_f = f0
        from optimizer_utils import report_best
        report_best(best_f, best_x)
        
        # main loop: global + local cycles
        while calls < budget:
            # global phase: random sampling with population size adaptive
            pop_size = max(1, min(5*dim, budget - calls))
            candidates = lb + self.rng.rand(pop_size, dim) * (ub - lb)
            for i in range(pop_size):
                if calls >= budget:
                    break
                x = candidates[i]
                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
            
            # local phase: pattern search on best point
            if calls >= budget:
                break
            # pattern search parameters
            step_size = 0.1 * (ub - lb)
            shrink = 0.5
            min_step = 1e-6 * (ub - lb)
            max_iter = min(50, budget - calls)
            x_curr = best_x.copy()
            f_curr = best_f
            for _ in range(max_iter):
                if calls >= budget:
                    break
                improved = False
                for axis in range(dim):
                    if calls >= budget:
                        break
                    # try positive direction
                    delta = np.zeros(dim)
                    delta[axis] = step_size[axis]
                    x_new = np.clip(x_curr + delta, lb, ub)
                    f_new = func(x_new)
                    calls += 1
                    if f_new < f_curr:
                        f_curr = f_new
                        x_curr = x_new.copy()
                        improved = True
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
                            report_best(best_f, best_x)
                        # after success, try to accelerate along same direction
                        if calls < budget:
                            x_new2 = np.clip(x_curr + delta, lb, ub)
                            f_new2 = func(x_new2)
                            calls += 1
                            if f_new2 < f_curr:
                                f_curr = f_new2
                                x_curr = x_new2.copy()
                                if f_new2 < best_f:
                                    best_f = f_new2
                                    best_x = x_new2.copy()
                                    report_best(best_f, best_x)
                    # if not improved, try negative direction
                    else:
                        x_new = np.clip(x_curr - delta, lb, ub)
                        f_new = func(x_new)
                        calls += 1
                        if f_new < f_curr:
                            f_curr = f_new
                            x_curr = x_new.copy()
                            improved = True
                            if f_new < best_f:
                                best_f = f_new
                                best_x = x_new.copy()
                                report_best(best_f, best_x)
                if not improved:
                    step_size = step_size * shrink
                if np.all(step_size < min_step):
                    break
        return best_f, best_x