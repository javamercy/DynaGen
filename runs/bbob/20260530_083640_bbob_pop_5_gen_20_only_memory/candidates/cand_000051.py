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

        # Step 1: initial point
        x_curr = lb + (ub - lb) * rng.rand(dim)
        f_curr = func(x_curr)
        evals = 1
        x_best = x_curr.copy()
        f_best = f_curr
        report_best(f_best, x_best)

        if evals >= budget:
            return f_best, x_best

        # Annealing parameters
        T0 = 1.0
        range_vec = ub - lb

        for e in range(evals, budget):
            frac = e / budget
            T = T0 * (1 - frac) ** 2
            step = range_vec * (1 - frac)
            cauchy_rnd = rng.standard_cauchy(dim)
            x_new = x_curr + step * cauchy_rnd
            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new)
            evals += 1

            if f_new < f_curr:
                accept = True
            else:
                delta = f_new - f_curr
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.uniform() < prob

            if accept:
                x_curr = x_new
                f_curr = f_new
                if f_curr < f_best:
                    f_best = f_curr
                    x_best = x_curr.copy()
                    report_best(f_best, x_best)

        return f_best, x_best