import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        rng = np.random.RandomState(self.seed)
        best_val = float('inf')
        best_x = None
        evals = 0

        x = rng.uniform(lb, ub)
        fx = func(x)
        evals += 1
        best_val = fx
        best_x = x.copy()
        report_best(best_val, best_x)

        step = 0.1 * (ub - lb)
        min_step = 1e-10 * (ub - lb)

        while evals < self.budget:
            improved = False
            for d in rng.permutation(dim):
                if evals >= self.budget:
                    break
                x_new = x.copy()
                x_new[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    f_new = func(x_new)
                    evals += 1
                    if f_new < best_val:
                        best_val = f_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                    if f_new < fx:
                        fx = f_new
                        x = x_new.copy()
                        improved = True
                        step[d] *= 1.2
                if not improved:
                    x_new = x.copy()
                    x_new[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                    if x_new[d] != x[d]:
                        f_new = func(x_new)
                        evals += 1
                        if f_new < best_val:
                            best_val = f_new
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                        if f_new < fx:
                            fx = f_new
                            x = x_new.copy()
                            improved = True
                            step[d] *= 1.2
                        else:
                            step[d] *= 0.5

            if improved:
                direction = x - best_x
                x_pattern = np.clip(x + direction, lb, ub)
                if evals < self.budget and np.any(x_pattern != x):
                    f_pattern = func(x_pattern)
                    evals += 1
                    if f_pattern < best_val:
                        best_val = f_pattern
                        best_x = x_pattern.copy()
                        report_best(best_val, best_x)
                    if f_pattern < fx:
                        fx = f_pattern
                        x = x_pattern.copy()
            else:
                step = np.clip(step * 0.5, min_step, None)
                if np.all(step <= min_step):
                    x = rng.uniform(lb, ub)
                    fx = func(x)
                    evals += 1
                    if fx < best_val:
                        best_val = fx
                        best_x = x.copy()
                        report_best(best_val, best_x)
                    step = 0.1 * (ub - lb)
                else:
                    x = best_x.copy()
                    fx = best_val

            if evals < self.budget and rng.rand() < 0.1:
                x_rand = rng.uniform(lb, ub)
                f_rand = func(x_rand)
                evals += 1
                if f_rand < best_val:
                    best_val = f_rand
                    best_x = x_rand.copy()
                    report_best(best_val, best_x)
                if f_rand < fx:
                    fx = f_rand
                    x = x_rand.copy()

        return best_val, best_x