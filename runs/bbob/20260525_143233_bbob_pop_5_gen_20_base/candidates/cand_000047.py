import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.window = max(10, 2 * dim)
        self.restart_threshold = max(10 * dim, 100)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        # initial point
        x = np.random.uniform(lb, ub, dim)
        fx = func(x)
        evals = 1
        best_x = x.copy()
        best_val = fx
        report_best(best_val, best_x)
        sigma = 0.2 * np.mean(ub - lb)
        successes = []
        no_improve_evals = 0
        while evals < budget:
            y = x + sigma * np.random.randn(dim)
            y = np.clip(y, lb, ub)
            fy = func(y)
            evals += 1
            if fy < fx:
                x = y
                fx = fy
                successes.append(True)
                if fy < best_val:
                    best_val = fy
                    best_x = y.copy()
                    report_best(best_val, best_x)
                no_improve_evals = 0
            else:
                successes.append(False)
                no_improve_evals += 1
            # step size adaptation every window evaluations
            if len(successes) >= self.window:
                recent = successes[-self.window:]
                success_rate = sum(recent) / len(recent)
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                successes = []
            # restart if stagnation
            if no_improve_evals >= self.restart_threshold:
                x = np.random.uniform(lb, ub, dim)
                fx = func(x)
                evals += 1
                if fx < best_val:
                    best_val = fx
                    best_x = x.copy()
                    report_best(best_val, best_x)
                sigma = 0.2 * np.mean(ub - lb)
                successes = []
                no_improve_evals = 0
        return best_val, best_x