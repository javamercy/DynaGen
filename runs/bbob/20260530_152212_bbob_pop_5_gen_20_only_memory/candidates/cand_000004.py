import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
    
    def __call__(self, func):
        low = func.bounds.lb
        high = func.bounds.ub
        dim = self.dim
        budget = self.budget
        
        # Initial point
        x0 = np.random.uniform(low, high, size=dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        report_best(best_f, best_x)
        evals = 1
        
        if budget <= 1:
            return best_f, best_x
        
        # Initial step size
        step = (high - low) * 0.1
        step = np.maximum(step, 1e-6 * np.ones_like(step))
        
        x = x0.copy()
        while evals < budget:
            improved = False
            for i in range(dim):
                if evals >= budget:
                    break
                # Try positive direction
                x_new = x.copy()
                x_new[i] += step[i]
                x_new = np.clip(x_new, low, high)
                f_new = func(x_new)
                evals += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    report_best(best_f, best_x)
                    x = x_new.copy()
                    improved = True
                    # Pattern step
                    if evals < budget:
                        x_pat = x_new.copy()
                        x_pat[i] += step[i]
                        x_pat = np.clip(x_pat, low, high)
                        f_pat = func(x_pat)
                        evals += 1
                        if f_pat < best_f:
                            best_f = f_pat
                            best_x = x_pat.copy()
                            report_best(best_f, best_x)
                            x = x_pat.copy()
                    break
                # Try negative direction
                x_new = x.copy()
                x_new[i] -= step[i]
                x_new = np.clip(x_new, low, high)
                f_new = func(x_new)
                evals += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    report_best(best_f, best_x)
                    x = x_new.copy()
                    improved = True
                    if evals < budget:
                        x_pat = x_new.copy()
                        x_pat[i] -= step[i]
                        x_pat = np.clip(x_pat, low, high)
                        f_pat = func(x_pat)
                        evals += 1
                        if f_pat < best_f:
                            best_f = f_pat
                            best_x = x_pat.copy()
                            report_best(best_f, best_x)
                            x = x_pat.copy()
                    break
            if not improved:
                step *= 0.5
        return best_f, best_x