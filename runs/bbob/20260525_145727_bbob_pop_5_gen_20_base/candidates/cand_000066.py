import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        # SPSA parameters
        a = 0.5
        A = 100
        alpha = 0.602
        c = 0.1
        gamma = 0.101

        # initial point
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        x = best_x.copy()
        k = 0
        # main loop: use 2 evals per iteration
        while evals + 2 <= budget:
            # perturbation vector
            delta = rng.choice([-1, 1], size=dim)
            # step sizes
            a_k = a / ((k + 1 + A) ** alpha)
            c_k = c / ((k + 1) ** gamma)

            # perturbed points
            x_plus = np.clip(x + c_k * delta, lb, ub)
            x_minus = np.clip(x - c_k * delta, lb, ub)

            f_plus = func(x_plus)
            f_minus = func(x_minus)
            evals += 2

            # gradient approximation
            g_hat = (f_plus - f_minus) / (2 * c_k * delta)

            # update
            x = x - a_k * g_hat
            x = np.clip(x, lb, ub)

            # evaluate new point
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

            k += 1

        # use remaining budget (if any) for local perturbations around best
        while evals < budget:
            # simple random perturbation
            perturbation = rng.randn(dim) * 0.1 * (ub - lb)
            candidate = np.clip(best_x + perturbation, lb, ub)
            val = func(candidate)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = candidate.copy()
                report_best(best_val, best_x)

        return best_val, best_x