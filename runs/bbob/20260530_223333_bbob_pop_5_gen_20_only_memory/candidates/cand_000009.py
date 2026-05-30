import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        
        # LHS initialization
        n_init = max(2, min(20, int(self.budget * 0.1)))
        lhs_points = np.zeros((n_init, dim))
        for i in range(dim):
            perm = self.rng.permutation(n_init)
            lhs_points[:, i] = (perm + self.rng.uniform(size=n_init)) / n_init
        lhs_points = lb + lhs_points * (ub - lb)
        
        best_value = np.inf
        best_x = None
        calls = 0
        
        for x in lhs_points:
            if calls >= self.budget:
                break
            val = func(x)
            calls += 1
            if val < best_value:
                best_value = val
                best_x = x.copy()
                report_best(best_value, best_x)
        
        sigma = np.mean(ub - lb) * 0.2
        center = best_x.copy()
        L = 10
        successes = []
        stall_counter = 0
        
        while calls < self.budget:
            x = center + self.rng.normal(0, sigma, size=dim)
            x = np.clip(x, lb, ub)
            val = func(x)
            calls += 1
            if val < best_value:
                best_value = val
                best_x = x.copy()
                report_best(best_value, best_x)
                center = x.copy()
                successes.append(1)
                stall_counter = 0
            else:
                successes.append(0)
                stall_counter += 1
            
            if len(successes) > L:
                successes.pop(0)
            
            if len(successes) == L:
                success_rate = sum(successes) / L
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
            
            if stall_counter >= 5 * L:
                x_restart = self.rng.uniform(lb, ub, size=dim)
                val_restart = func(x_restart)
                calls += 1
                if val_restart < best_value:
                    best_value = val_restart
                    best_x = x_restart.copy()
                    report_best(best_value, best_x)
                center = x_restart.copy()
                sigma = np.mean(ub - lb) * 0.2
                stall_counter = 0
                successes = []
        
        return (best_value, best_x)