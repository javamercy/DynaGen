import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        calls = 0
        best_x = lb + self.rng.rand(self.dim) * (ub - lb)
        best_val = func(best_x)
        calls += 1
        report_best(best_val, best_x)
        step = 0.1 * (ub - lb)
        improvement = True
        while calls < self.budget:
            if improvement or self.rng.rand() < 0.1:
                for d in range(self.dim):
                    if calls >= self.budget:
                        break
                    x_candidate = np.clip(best_x + step * np.eye(self.dim)[d], lb, ub)
                    val = func(x_candidate)
                    calls += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_candidate
                        step[d] *= 2.0
                        report_best(best_val, best_x)
                        improvement = True
                        break
                    x_candidate = np.clip(best_x - step * np.eye(self.dim)[d], lb, ub)
                    val = func(x_candidate)
                    calls += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_candidate
                        step[d] *= 2.0
                        report_best(best_val, best_x)
                        improvement = True
                        break
                    step[d] *= 0.5
                else:
                    improvement = False
            else:
                # random perturbation
                delta = self.rng.randn(self.dim) * 0.1 * step.mean()
                x_candidate = np.clip(best_x + delta, lb, ub)
                val = func(x_candidate)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x_candidate
                    report_best(best_val, best_x)
                    improvement = True
                else:
                    improvement = False
        return best_val, best_x