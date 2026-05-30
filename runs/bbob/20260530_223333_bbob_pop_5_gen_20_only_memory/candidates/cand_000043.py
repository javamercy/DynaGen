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

        best_val = np.inf
        best_x = None

        def evaluate(x):
            nonlocal best_val, best_x
            val = func(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            return val

        n_restarts = max(1, budget // (10 * dim))
        n_restarts = min(n_restarts, 10)
        evals_per_restart = budget // n_restarts
        leftover = budget - n_restarts * evals_per_restart

        for restart in range(n_restarts):
            evals_used = 0
            if best_x is not None and rng.rand() < 0.5:
                x = best_x.copy() + 0.05 * rng.uniform(-1, 1, dim) * (ub - lb)
                x = np.clip(x, lb, ub)
            else:
                x = rng.uniform(lb, ub)
            f_x = evaluate(x)
            evals_used += 1
            local_best_val = f_x
            local_best_x = x.copy()
            sigma = 0.1 * (ub - lb)
            min_sigma = 1e-12 * (ub - lb)
            no_improve = 0
            max_no_improve = max(2 * dim, 10)

            while evals_used < evals_per_restart and no_improve < max_no_improve:
                z = rng.randn(dim)
                candidate = x + sigma * z
                candidate = np.clip(candidate, lb, ub)
                if np.all(candidate == x):
                    continue
                val = evaluate(candidate)
                evals_used += 1
                if val < f_x:
                    x = candidate
                    f_x = val
                    if val < local_best_val:
                        local_best_val = val
                        local_best_x = x.copy()
                    no_improve = 0
                    # increase step size
                    sigma *= np.exp(1.0 / (2.0 * dim))
                else:
                    no_improve += 1
                    # decrease step size
                    sigma *= np.exp(-0.5 / dim)
                sigma = np.maximum(sigma, min_sigma)
                # if step sizes become too small, break
                if np.all(sigma <= min_sigma):
                    break

            # fill remaining evaluations with random sampling
            while evals_used < evals_per_restart:
                x_rand = rng.uniform(lb, ub)
                evaluate(x_rand)
                evals_used += 1

        # use leftover evaluations
        for _ in range(leftover):
            x_rand = rng.uniform(lb, ub)
            evaluate(x_rand)

        return best_val, best_x